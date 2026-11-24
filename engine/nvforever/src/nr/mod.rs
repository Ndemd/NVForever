//! DLSS 5 Neural Rendering on the engine's own frame (the OpenDLSS-NR network, MIT, in `nr.dll`
//! the player supplies with the extracted weights). The network runs at the render resolution,
//! before DLSS Super Resolution upscales: the scene color is encoded to the display proxy the
//! network reads, the network's head is composed back onto the scene, and DLSS then upscales the
//! neural-rendered frame. The HUD is not drawn yet at this point (Bevy UI comes after the
//! upscaler), so it is untouched by design.
//!
//! Prototype gates: Windows + Vulkan, the `nr` folder beside the program (`NVF_NR_DIR` overrides)
//! holding `nr.dll`, `model/` (full or small network), `shaders/`, `ptx/`; `NVF_NR=0` turns it off.
//! Single-frame mode: the temporal history path is a later phase.
//!
//! The raw-Vulkan recording follows `dlss::evaluate` exactly: a dedicated encoder touched only
//! through `as_hal_mut`, full barriers around the foreign work, `ctx.add_command_buffer`, and the
//! textures/buffers kept alive until the submitted work is done.

mod ffi;
pub mod overlay;
mod shaders;

use std::ffi::CString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex, PoisonError};

use ash::vk;
use bevy::log::Level;
use bevy::prelude::*;
use bevy::render::renderer::raw_vulkan_init::{AdditionalVulkanFeatures, RawVulkanInitSettings};
use bevy::render::renderer::{RenderContext, RenderDevice, ViewQuery};
use bevy::render::view::ViewTarget;
use bevy::render::{Extract, RenderApp};
use bevy::core_pipeline::Core3dSystems;

use crate::upscaler::prep::{PreparedUpscale, UpscaleSettings};

/// `NVF_NR=0` turns the network off; anything else (or unset) lets it run when the folder is there.
static ALLOWED: LazyLock<bool> = LazyLock::new(|| !std::env::var("NVF_NR").is_ok_and(|v| v.trim() == "0"));
/// The proxy's paper white in scene units (`NVF_NR_WHITE`, default 1.0): the HDR scene is divided
/// by it before the network's LDR proxy. Tune to the game's lighting scale.
static WHITE: LazyLock<f32> = LazyLock::new(|| std::env::var("NVF_NR_WHITE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1.0));
/// The composite's intensity (`NVF_NR_INTENSITY`, default 1.0): 0 leaves the scene as it was.
static INTENSITY: LazyLock<f32> = LazyLock::new(|| std::env::var("NVF_NR_INTENSITY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1.0));
/// The network's style conditioning and the composite's grade (`NVF_NR_STYLE`): 0 off (Reference),
/// 1 Natural, 2 Cinematic — as the native runtime's DLSSNR.Style and RenoDX's presets.
static STYLE: LazyLock<f32> = LazyLock::new(|| std::env::var("NVF_NR_STYLE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0.0));
/// localTone conditioning (`NVF_NR_TONE`, default 1.0, range 0..4): also scales the style grade.
static TONE: LazyLock<f32> = LazyLock::new(|| std::env::var("NVF_NR_TONE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1.0));
/// localStructure conditioning (`NVF_NR_STRUCTURE`, default 1.0, range 0..4): detail strength.
static STRUCTURE: LazyLock<f32> = LazyLock::new(|| std::env::var("NVF_NR_STRUCTURE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1.0));
/// skinStructure conditioning (`NVF_NR_SKIN`, default -1.0 = follow localStructure; -1..4).
static SKIN: LazyLock<f32> = LazyLock::new(|| std::env::var("NVF_NR_SKIN").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(-1.0));
/// autoMask (`NVF_NR_AUTOMASK=0` turns off; default on): the network's own skin/detail masking.
static AUTOMASK: LazyLock<bool> = LazyLock::new(|| !std::env::var("NVF_NR_AUTOMASK").is_ok_and(|v| v.trim() == "0"));
/// `NVF_NR_NETWORK=full` runs the 71-block network; the default is the small one (31 blocks).
static NETWORK: LazyLock<String> = LazyLock::new(|| std::env::var("NVF_NR_NETWORK").unwrap_or_else(|_| "small".into()));

/// The folder beside the program (or `NVF_NR_DIR`): nr.dll, model/, shaders/, ptx/.
fn folder() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("NVF_NR_DIR") {
        let p = PathBuf::from(dir);
        return p.join("nr.dll").is_file().then_some(p);
    }
    let exe = std::env::current_exe().ok()?;
    let p = exe.parent()?.join("nr");
    p.join("nr.dll").is_file().then_some(p)
}

fn note(level: Level, line: impl AsRef<str>) {
    match level {
        Level::WARN => bevy::log::warn!("NR: {}", line.as_ref()),
        Level::ERROR => bevy::log::error!("NR: {}", line.as_ref()),
        _ => bevy::log::info!("NR: {}", line.as_ref()),
    }
}

/// The loaded DLL's API, taken at start (before the renderer).
static API: Mutex<Option<ffi::Api>> = Mutex::new(None);
/// The device callback enabled the network's extensions and features.
static DEVICE_READY: AtomicBool = AtomicBool::new(false);
/// The pass drew the last frame (the overlay's status line).
pub(crate) static NR_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Marks, in Bevy's `AdditionalVulkanFeatures`, a device made with the network's requirements.
struct NrDeviceEnabled;

/// Added by `main` before `DefaultPlugins` (as `DlssInitPlugin` is): loads nr.dll and asks for the
/// device extensions/features the network needs.
pub struct NrInitPlugin;

impl Plugin for NrInitPlugin {
    fn build(&self, app: &mut App) {
        // RivaTuner's and ReShade's Vulkan layers crash device creation with the network's
        // feature structs; keep them out of this process unless the player chose layers
        // themselves. Set before DefaultPlugins creates the Vulkan instance.
        if std::env::var_os("VK_LOADER_LAYERS_DISABLE").is_none() {
            std::env::set_var("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_RTSS,VK_LAYER_reshade");
        }
        if !cfg!(windows) || !*ALLOWED {
            return;
        }
        let Some(folder) = folder() else {
            note(Level::INFO, "inactive: no nr folder with nr.dll beside the program (NVF_NR_DIR names one)");
            return;
        };
        // SAFETY: the DLL stays loaded for the process; the API is only called after the device exists.
        match unsafe { ffi::load(&folder) } {
            Ok(api) => {
                *API.lock().unwrap_or_else(PoisonError::into_inner) = Some(api);
                note(Level::INFO, format!("nr.dll loaded from {}", folder.display()));
            }
            Err(line) => {
                note(Level::WARN, format!("inactive: {line}"));
                return;
            }
        }
        let mut settings = app.world_mut().get_resource_or_init::<RawVulkanInitSettings>();
        // SAFETY: the callback only adds extensions the adapter offers and merges features on
        // (never off), as nv_low_latency's does.
        unsafe {
            settings.add_create_device_callback(|args, adapter, features| {
                let caps = adapter.physical_device_capabilities();
                for extension in ffi::EXTENSIONS {
                    if !caps.supports_extension(extension) {
                        note(Level::WARN, format!("inactive: the driver has no {}", extension.to_string_lossy()));
                        return;
                    }
                }
                for extension in ffi::EXTENSIONS {
                    if !args.extensions.contains(&extension) {
                        args.extensions.push(extension);
                    }
                }
                ffi::merge_device_features(args.create_info);
                features.insert::<NrDeviceEnabled>();
                DEVICE_READY.store(true, Ordering::Relaxed);
            });
        }
    }

    /// After the renderer's device exists: open the network on it (as `dlss::win::after_device`).
    fn cleanup(&self, app: &mut App) {
        if !DEVICE_READY.load(Ordering::Relaxed) {
            return;
        }
        let Some(api) = *API.lock().unwrap_or_else(PoisonError::into_inner) else { return };
        let Some(folder) = folder() else { return };
        let Some(render) = app.get_sub_app_mut(RenderApp) else { return };
        let world = render.world_mut();
        let marker = world.get_resource::<AdditionalVulkanFeatures>().is_some_and(|f| f.has::<NrDeviceEnabled>());
        if !marker {
            return;
        }
        let (Some(device), Some(adapter)) = (world.get_resource::<RenderDevice>(), world.get_resource::<bevy::render::renderer::RenderAdapter>()) else {
            return;
        };
        // SAFETY: only read; the handles are cloned out of the guards.
        let (Some(hal_device), Some(hal_adapter)) = (unsafe { device.wgpu_device().as_hal::<wgpu::hal::api::Vulkan>() }, unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }) else {
            note(Level::WARN, "inactive: the renderer is not Vulkan");
            return;
        };
        let raw_device = hal_device.raw_device().clone();
        let instance = hal_adapter.shared_instance().raw_instance().handle();
        let physical = hal_adapter.raw_physical_device();
        let family = hal_device.queue_family_index();
        let queue_index = hal_device.queue_index();
        drop(hal_device);
        drop(hal_adapter);

        let model = folder.join("model").join(NETWORK.as_str());
        let cstrings: Vec<CString> = [model.clone(), folder.join("shaders"), folder.join("ptx")]
            .iter()
            .map(|p| CString::new(p.as_os_str().as_encoded_bytes()).unwrap_or_default())
            .collect();
        let mut err = [0i8; 1024];
        // SAFETY: the raw handles are the renderer's own, alive for the app; the paths live in
        // cstrings through the call; nr_open copies what it keeps.
        let handle = unsafe {
            (api.open)(
                instance,
                physical,
                raw_device.handle(),
                family,
                queue_index,
                cstrings[0].as_ptr(),
                cstrings[1].as_ptr(),
                cstrings[2].as_ptr(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if handle.is_null() {
            let line = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy();
            note(Level::WARN, format!("inactive: nr_open failed: {line}"));
            return;
        }
        let blocks = unsafe { (api.block_count)(handle) };
        let blend = unsafe { (api.blend_scale)(handle) };
        note(Level::INFO, format!("network open: {blocks} blocks, blend_scale {blend:.4}, model {}", model.display()));
        world.insert_resource(NrState {
            api,
            handle,
            folder,
            instance,
            physical,
            device: raw_device.handle(),
            family,
            queue_index,
            network: NETWORK.clone(),
            size: UVec2::ZERO,
            full: UVec2::ZERO,
            proxy: None,
            head_copy: None,
            encode_params: None,
            composite_params: None,
            encode_pipeline: None,
            composite_pipeline: None,
            history: Vec::new(),
            history_views: Vec::new(),
            parity: 0,
            history_primed: false,
            prev_jitter: Vec2::ZERO,
            features: None,
            preprocess_pipeline: None,
            preprocess_params: None,
            sampler: None,
            dummy_motion: None,
            dummy_motion_view: None,
            blend_scale: unsafe { (api.blend_scale)(handle) },
            frame: 0,
            failed: false,
        });
    }
}

/// The render-world state: the open network and the per-size resources around it.
#[derive(Resource)]
struct NrState {
    api: ffi::Api,
    handle: *mut std::ffi::c_void,
    folder: PathBuf,
    /// The renderer's raw handles, kept for a live network switch (small <-> full).
    instance: vk::Instance,
    physical: vk::PhysicalDevice,
    device: vk::Device,
    family: u32,
    queue_index: u32,
    network: String,
    size: UVec2,
    full: UVec2,
    proxy: Option<wgpu::Buffer>,
    head_copy: Option<wgpu::Buffer>,
    encode_params: Option<wgpu::Buffer>,
    composite_params: Option<wgpu::Buffer>,
    encode_pipeline: Option<wgpu::ComputePipeline>,
    composite_pipeline: Option<wgpu::ComputePipeline>,
    /// Temporal path: ping-pong history surfaces (proxy code space, half grid), the parity of
    /// the next write, and whether a history exists at all yet.
    history: Vec<wgpu::Texture>,
    history_views: Vec<wgpu::TextureView>,
    parity: u32,
    history_primed: bool,
    prev_jitter: Vec2,
    features: Option<wgpu::Buffer>,
    preprocess_pipeline: Option<wgpu::ComputePipeline>,
    preprocess_params: Option<wgpu::Buffer>,
    sampler: Option<wgpu::Sampler>,
    /// A zero motion texture for frames without a prepass (temporal path off then).
    dummy_motion: Option<wgpu::Texture>,
    dummy_motion_view: Option<wgpu::TextureView>,
    blend_scale: f32,
    frame: u32,
    failed: bool,
}

// SAFETY: the handle is only used on the render thread, where this resource lives; the DLL's
// own state is render-thread-only by the same contract as Streamline's.
unsafe impl Send for NrState {}
unsafe impl Sync for NrState {}

/// The pass plugin (added by `GraphicsPlugin`): the network between the occlusion and the upscaler.
pub struct NrPassPlugin;

/// The settings as the render world sees them (extracted every frame from the overlay's resource).
#[derive(Resource, Clone, Default)]
struct NrSettingsRender {
    enabled: bool,
    style: f32,
    intensity: f32,
    tone: f32,
    structure: f32,
    skin: f32,
    automask: bool,
    white: f32,
    network: String,
    temporal: bool,
}

fn extract_settings(mut commands: Commands, s: Extract<Res<overlay::NrSettings>>) {
    commands.insert_resource(NrSettingsRender {
        enabled: s.enabled,
        style: s.style,
        intensity: s.intensity,
        tone: s.tone,
        structure: s.structure,
        skin: s.skin,
        automask: s.automask,
        white: s.white,
        network: s.network.clone(),
        temporal: s.temporal,
    });
}

impl Plugin for NrPassPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else { return };
        render_app
            .init_resource::<NrSettingsRender>()
            .add_systems(bevy::render::ExtractSchedule, extract_settings)
            .add_systems(
                bevy::core_pipeline::Core3d,
                nr_pass
                    .in_set(Core3dSystems::EarlyPostProcess)
                    .after(crate::ssao::ao_before_fsr)
                    .before(crate::dlss::dlss_pass)
                    .before(crate::fsr::fsr_pass),
            );
    }
}

/// Switches the open network (small <-> full) live: close, open the other model directory, and
/// let the next frame rebuild the per-size resources.
fn reopen(state: &mut NrState, network: &str) -> bool {
    let model = state.folder.join("model").join(network);
    let cstrings: Vec<CString> = [model.clone(), state.folder.join("shaders"), state.folder.join("ptx")]
        .iter()
        .map(|p| CString::new(p.as_os_str().as_encoded_bytes()).unwrap_or_default())
        .collect();
    let mut err = [0i8; 1024];
    // SAFETY: the old handle is closed (its own queue drained) before the new one opens on the
    // same renderer handles, which stay alive for the app.
    unsafe {
        (state.api.close)(state.handle);
        let handle = (state.api.open)(
            state.instance,
            state.physical,
            state.device,
            state.family,
            state.queue_index,
            cstrings[0].as_ptr(),
            cstrings[1].as_ptr(),
            cstrings[2].as_ptr(),
            err.as_mut_ptr(),
            err.len(),
        );
        if handle.is_null() {
            let line = std::ffi::CStr::from_ptr(err.as_ptr()).to_string_lossy();
            note(Level::WARN, format!("network switch to {network} failed: {line}"));
            state.handle = std::ptr::null_mut();
            return false;
        }
        state.handle = handle;
    }
    state.network = network.to_string();
    state.size = UVec2::ZERO;
    state.full = UVec2::ZERO;
    state.proxy = None;
    state.head_copy = None;
    state.history.clear();
    state.history_views.clear();
    state.parity = 0;
    state.history_primed = false;
    state.features = None;
    state.preprocess_pipeline = None;
    state.blend_scale = unsafe { (state.api.blend_scale)(state.handle) };
    note(Level::INFO, format!("network switched to {network}"));
    true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EncodeParams {
    width: u32,
    height: u32,
    paper_white: f32,
    exposure: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PreprocessParams {
    full_width: u32,
    full_height: u32,
    valid_width: u32,
    valid_height: u32,
    seed: u32,
    history_valid: u32,
    style: f32,
    tone: f32,
    structure: f32,
    skin: f32,
    automask: f32,
    jitter_dx: f32,
    jitter_dy: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CompositeParams {
    width: u32,
    height: u32,
    full_width: u32,
    full_height: u32,
    paper_white: f32,
    exposure: f32,
    intensity: f32,
    style: f32,
    local_tone: f32,
    blend_scale: f32,
    jitter_dx: f32,
    jitter_dy: f32,
    history_valid: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

fn bytes_of<T: Copy>(value: &T) -> &[u8] {
    // SAFETY: the params structs are repr(C) of u32/f32 only, no padding beyond the explicit one.
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>()) }
}

/// Rebuilds the per-size buffers and pipelines (once per resolution).
fn prepare_size(state: &mut NrState, device: &wgpu::Device, queue: &wgpu::Queue, size: UVec2) -> bool {
    let mut err = [0i8; 1024];
    // SAFETY: the handle is open; resize drains the library's own queue work first.
    let ok = unsafe { (state.api.resize)(state.handle, size.x, size.y, err.as_mut_ptr(), err.len()) };
    if ok == 0 {
        let line = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy();
        note(Level::WARN, format!("nr_resize failed: {line}"));
        return false;
    }
    let full = UVec2::new(unsafe { (state.api.full_width)(state.handle) }, unsafe { (state.api.full_height)(state.handle) });
    let proxy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nr_proxy"),
        size: (size.x * size.y * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let head_copy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nr_head_copy"),
        size: (full.x * full.y * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let encode_params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nr_encode_params"),
        size: std::mem::size_of::<EncodeParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let composite_params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nr_composite_params"),
        size: std::mem::size_of::<CompositeParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let encode_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("nr_encode"), source: wgpu::ShaderSource::Wgsl(shaders::encode_source().into()) });
    let composite_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("nr_composite"), source: wgpu::ShaderSource::Wgsl(shaders::composite_source().into()) });
    let preprocess_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("nr_preprocess"), source: wgpu::ShaderSource::Wgsl(shaders::preprocess_source().into()) });
    let encode_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("nr_encode"),
        layout: None,
        module: &encode_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let composite_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("nr_composite"),
        layout: None,
        module: &composite_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let preprocess_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("nr_preprocess"),
        layout: None,
        module: &preprocess_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let preprocess_params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nr_preprocess_params"),
        size: std::mem::size_of::<PreprocessParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let features = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nr_features"),
        size: (full.x * full.y * 64) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("nr_history"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..default()
    });
    let mut history = Vec::new();
    let mut history_views = Vec::new();
    for i in 0..2 {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(if i == 0 { "nr_history_a" } else { "nr_history_b" }),
            size: wgpu::Extent3d { width: size.x, height: size.y, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        history_views.push(tex.create_view(&default()));
        history.push(tex);
    }
    let dummy_motion = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("nr_dummy_motion"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let dummy_motion_view = dummy_motion.create_view(&default());
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &dummy_motion, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &[0u8; 4],
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    state.size = size;
    state.full = full;
    state.proxy = Some(proxy);
    state.head_copy = Some(head_copy);
    state.encode_params = Some(encode_params);
    state.composite_params = Some(composite_params);
    state.encode_pipeline = Some(encode_pipeline);
    state.composite_pipeline = Some(composite_pipeline);
    state.preprocess_pipeline = Some(preprocess_pipeline);
    state.preprocess_params = Some(preprocess_params);
    state.features = Some(features);
    state.sampler = Some(sampler);
    state.history = history;
    state.history_views = history_views;
    state.parity = 0;
    state.history_primed = false;
    state.dummy_motion = Some(dummy_motion);
    state.dummy_motion_view = Some(dummy_motion_view);
    note(Level::INFO, format!("sized {size} -> field {full}"));
    true
}

/// The network on the DLSS camera's render rect, before the upscaler (the module comment).
/// Temporal path: the preprocess reprojects the previous neural output with the engine's motion
/// vectors and jitter delta; the composite blends by the network's own logit and publishes the
/// next history (half grid, truncated).
fn nr_pass(
    view: ViewQuery<(&ViewTarget, &PreparedUpscale, Option<&bevy::core_pipeline::prepass::ViewPrepassTextures>, &bevy::render::camera::TemporalJitter)>,
    mut state: Option<ResMut<NrState>>,
    settings: Res<NrSettingsRender>,
    device: Res<RenderDevice>,
    queue: Res<bevy::render::renderer::RenderQueue>,
    mut ctx: RenderContext,
) {
    let Some(state) = state.as_mut() else { return };
    if state.failed || state.handle.is_null() {
        return;
    }
    if !settings.enabled {
        NR_ACTIVE.store(false, Ordering::Relaxed);
        return;
    }
    if settings.network != state.network && !reopen(state, &settings.network) {
        state.failed = true;
        return;
    }
    let (target, prepared, prepass, jitter) = view.into_inner();
    let UpscaleSettings::Dlss(_) = prepared.camera.settings else {
        NR_ACTIVE.store(false, Ordering::Relaxed);
        return;
    };
    let size = prepared.render_size;
    if size.x == 0 || size.y == 0 {
        return;
    }
    let device = device.wgpu_device();
    if state.size != size || state.proxy.is_none() {
        if !prepare_size(state, device, &queue, size) {
            state.failed = true;
            return;
        }
    }
    let (Some(proxy), Some(head_copy), Some(encode_params), Some(composite_params), Some(encode_pipeline), Some(composite_pipeline), Some(preprocess_pipeline), Some(preprocess_params), Some(features), Some(sampler)) =
        (&state.proxy, &state.head_copy, &state.encode_params, &state.composite_params, &state.encode_pipeline, &state.composite_pipeline, &state.preprocess_pipeline, &state.preprocess_params, &state.features, &state.sampler)
    else {
        return;
    };
    let color_view = target.main_texture_view();
    let seed = if prepared.reset { 0 } else { state.frame };
    let motion_view = prepass
        .and_then(|p| p.motion_vectors.as_ref())
        .map(|m| std::ops::Deref::deref(&m.texture.default_view))
        .unwrap_or(state.dummy_motion_view.as_ref().expect("sized"));
    // TemporalJitter.offset is minus the applied jitter (the engine passes -jitter to
    // Streamline), so (j_prev - j_cur) = (offset_cur - offset_prev).
    let jitter_delta = (jitter.offset - state.prev_jitter) / size.as_vec2();
    let history_valid = u32::from(state.history_primed && !prepared.reset && settings.temporal);
    let prev_parity = (state.parity ^ 1) as usize;
    let next_parity = state.parity as usize;

    // 1. Encode the scene's render rect to the proxy, on Bevy's encoder.
    let encode = EncodeParams { width: size.x, height: size.y, paper_white: settings.white, exposure: 1.0 };
    queue.write_buffer(encode_params, 0, bytes_of(&encode));
    {
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nr_encode"),
            layout: &encode_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: proxy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: encode_params.as_entire_binding() },
            ],
        });
        let encoder = ctx.command_encoder();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("nr_encode"), timestamp_writes: None });
        pass.set_pipeline(encode_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((size.x + 7) / 8, (size.y + 7) / 8, 1);
    }

    // 2. The temporal preprocess: 16 feature lanes per field pixel, history reprojected with
    // the engine MVs + jitter delta, on Bevy's encoder.
    let full = state.full;
    let pre = PreprocessParams {
        full_width: full.x,
        full_height: full.y,
        valid_width: size.x,
        valid_height: size.y,
        seed,
        history_valid,
        style: settings.style,
        tone: settings.tone,
        structure: settings.structure,
        skin: settings.skin,
        automask: if settings.automask { 1.0 } else { -1.0 },
        jitter_dx: jitter_delta.x,
        jitter_dy: jitter_delta.y,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    queue.write_buffer(preprocess_params, 0, bytes_of(&pre));
    {
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nr_preprocess"),
            layout: &preprocess_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: proxy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(motion_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&state.history_views[prev_parity]) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: features.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: preprocess_params.as_entire_binding() },
            ],
        });
        let encoder = ctx.command_encoder();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("nr_preprocess"), timestamp_writes: None });
        pass.set_pipeline(preprocess_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((full.x + 7) / 8, (full.y + 7) / 8, 1);
    }

    // 3. The network on its own encoder, raw recording only (dlss::evaluate's pattern).
    let raw_features = unsafe { features.as_hal::<wgpu::hal::api::Vulkan>() }.map(|b| unsafe { b.raw_handle() });
    let raw_head_copy = unsafe { head_copy.as_hal::<wgpu::hal::api::Vulkan>() }.map(|b| unsafe { b.raw_handle() });
    let raw_device = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.map(|d| d.raw_device().clone());
    let (Some(raw_features), Some(raw_head_copy), Some(raw_device)) = (raw_features, raw_head_copy, raw_device) else {
        state.failed = true;
        return;
    };
    let api = state.api;
    let handle = state.handle;
    let mut nr_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nr_network") });
    // SAFETY: the closure records only through the raw command buffer wgpu opened; the encoder is
    // not touched otherwise meanwhile; the handles stay alive through the kept closure below.
    let recorded = unsafe {
        nr_encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|hal| {
            hal.map(|hal| {
                let cmd = hal.raw_handle();
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
                let full_barrier = || {
                    raw_device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &barrier, &[], &[]);
                };
                full_barrier();
                (api.reset_pools)(handle);
                let nr_features = (api.features)(handle);
                let region = vk::BufferCopy::default().size((full.x * full.y * 64) as u64);
                raw_device.cmd_copy_buffer(cmd, raw_features, nr_features, &[region]);
                full_barrier();
                let mut err = [0i8; 1024];
                let ok = (api.record)(handle, cmd, err.as_mut_ptr(), err.len());
                if ok == 0 {
                    return false;
                }
                full_barrier();
                let head = (api.head)(handle);
                let region = vk::BufferCopy::default().size((full.x * full.y * 16) as u64);
                raw_device.cmd_copy_buffer(cmd, head, raw_head_copy, &[region]);
                full_barrier();
                true
            })
        })
    };
    let command_buffer = nr_encoder.finish();
    let kept = (target.main_texture().clone(), proxy.clone(), head_copy.clone(), features.clone());
    command_buffer.on_submitted_work_done(move || drop(kept));
    ctx.add_command_buffer(command_buffer);
    if recorded != Some(true) {
        note(Level::WARN, "the network's recording failed; NR is off for the session");
        state.failed = true;
        return;
    }
    if unsafe { (api.chain_timeouts)(handle) } != 0 {
        note(Level::WARN, "chain timeouts; disabling counter chaining");
        unsafe { (api.set_chain)(0) };
    }

    // 4. Compose the head back onto the scene's render rect and publish the next history,
    // on Bevy's next encoder.
    let composite = CompositeParams {
        width: size.x,
        height: size.y,
        full_width: full.x,
        full_height: full.y,
        paper_white: settings.white,
        exposure: 1.0,
        intensity: settings.intensity,
        style: settings.style,
        local_tone: settings.tone,
        blend_scale: state.blend_scale,
        jitter_dx: jitter_delta.x,
        jitter_dy: jitter_delta.y,
        history_valid,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    queue.write_buffer(composite_params, 0, bytes_of(&composite));
    {
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nr_composite"),
            layout: &composite_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: head_copy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: proxy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&state.history_views[prev_parity]) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&state.history_views[next_parity]) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(motion_view) },
                wgpu::BindGroupEntry { binding: 7, resource: composite_params.as_entire_binding() },
            ],
        });
        let encoder = ctx.command_encoder();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("nr_composite"), timestamp_writes: None });
        pass.set_pipeline(composite_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((size.x + 7) / 8, (size.y + 7) / 8, 1);
    }
    state.parity ^= 1;
    state.history_primed = true;
    state.prev_jitter = jitter.offset;
    state.frame = state.frame.wrapping_add(1);
    NR_ACTIVE.store(true, Ordering::Relaxed);
}
