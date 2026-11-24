//! The two compute shaders around the network: the HDR scene to the display proxy the network
//! reads, and the head's answer back onto the scene. The transforms are the demo's
//! (`OpenDLSS-NR/docs/frame.md`): white divide, the soft shoulder above 0.75 (K = 5.770780),
//! sRGB, the half grid; the composite inverts them. Single-frame prototype: no history blend yet
//! (the network's blend logit is read but the temporal path is a later phase).

/// f32 -> half -> f32 round-to-nearest-even for values in the normal half range (the proxy is
/// 0..1): the classic mantissa rounding, exact enough for the display proxy (the bit-exact
/// publication matters for parity fixtures, not for the picture).
const ROUND_F16: &str = r#"
fn round_f16(x: f32) -> f32 {
    let u = bitcast<u32>(x);
    let rounded = (u + 0x0fffu + ((u >> 13u) & 1u)) & 0xffffe000u;
    return bitcast<f32>(rounded);
}
fn round_f16_3(v: vec3<f32>) -> vec3<f32> {
    return vec3(round_f16(v.x), round_f16(v.y), round_f16(v.z));
}
"#;

pub const ENCODE: &str = r#"
struct Params {
    width: u32,
    height: u32,
    paper_white: f32,
    exposure: f32,
}
@group(0) @binding(0) var color: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> proxy: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn srgb_encode(v: vec3<f32>) -> vec3<f32> {
    let lo = 12.92 * v;
    let hi = 1.055 * pow(max(v, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return select(lo, hi, v > vec3(0.0031308));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) { return; }
    let scene = max(textureLoad(color, vec2<i32>(id.xy), 0).rgb, vec3(0.0));
    var v = scene * params.exposure / max(params.paper_white, 0.05);
    let over = v > vec3(0.75);
    let knee = vec3(0.75) + 0.25 * (vec3(1.0) - exp(-5.770780 * (v - vec3(0.75))));
    v = select(v, knee, over);
    let code = round_f16_3(srgb_encode(clamp(v, vec3(0.0), vec3(1.0))));
    proxy[id.y * params.width + id.x] = vec4(code, 1.0);
}
"#;

pub const COMPOSITE: &str = r#"
struct Params {
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
@group(0) @binding(0) var out_color: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(1) var<storage, read> head: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> proxy: array<vec4<f32>>;
@group(0) @binding(3) var history_tex: texture_2d<f32>;
@group(0) @binding(4) var lin_sampler: sampler;
@group(0) @binding(5) var history_next: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var motion_tex: texture_2d<f32>;
@group(0) @binding(7) var<uniform> params: Params;

fn srgb_decode(v: vec3<f32>) -> vec3<f32> {
    let lo = v / 12.92;
    let hi = pow((v + 0.055) / 1.055, vec3(2.4));
    return select(lo, hi, v > vec3(0.04045));
}

// The native runtime's Natural/Cinematic grade (OpenDLSS-NR demo nr_composite.comp, MIT):
// per-channel exposure + smoothstep contrast in proxy code space, then a saturation scale in HSL.
fn nr_hsl(rgb: vec3<f32>) -> vec3<f32> {
    let high = max(max(rgb.r, rgb.g), rgb.b);
    let low = min(min(rgb.r, rgb.g), rgb.b);
    let sum = high + low;
    let light = sum * 0.5;
    var hue = 0.0;
    var saturation = 0.0;
    if (high > low) {
        let delta = high - low;
        saturation = select(delta / sum, delta / ((2.0 - high) - low), light > 0.5);
        if (high == rgb.r) {
            hue = ((rgb.g - rgb.b) / delta + select(0.0, 6.0, rgb.g < rgb.b)) / 6.0;
        } else if (high == rgb.g) {
            hue = ((rgb.b - rgb.r) / delta + 2.0) / 6.0;
        } else {
            hue = ((rgb.r - rgb.g) / delta + 4.0) / 6.0;
        }
    }
    return vec3(hue, saturation, light);
}

fn nr_hue(p: f32, q: f32, hue_in: f32) -> f32 {
    var h = hue_in;
    if (h < 0.0) { h += 1.0; }
    if (h > 1.0) { h -= 1.0; }
    if (h < 0.16666667) { return p + (q - p) * 6.0 * h; }
    if (h < 0.5) { return q; }
    if (h < 0.6666667) { return p + (q - p) * (0.6666667 - h) * 6.0; }
    return p;
}

fn nr_rgb(hsl: vec3<f32>) -> vec3<f32> {
    if (hsl.y <= 0.0) { return vec3(hsl.z); }
    let q = select(hsl.z + hsl.y - hsl.z * hsl.y, hsl.z * (hsl.y + 1.0), hsl.z < 0.5);
    let p = 2.0 * hsl.z - q;
    return vec3(nr_hue(p, q, hsl.x + 0.33333334), nr_hue(p, q, hsl.x), nr_hue(p, q, hsl.x - 0.33333334));
}

fn nr_style(neural: vec3<f32>, style: f32, tone: f32) -> vec3<f32> {
    let style_tone = clamp(tone, 0.0, 1.0);
    let exposure = select(0.0, -0.1 * style_tone, style == 1.0);
    let contrast = select(0.0, -0.25 * style_tone, style == 1.0);
    let saturation = select(-0.15 * style_tone, -0.1 * style_tone, style == 1.0);
    var color = vec3(0.0);
    for (var c = 0u; c < 3u; c++) {
        let value = clamp(neural[c], 0.0, 1.0);
        let exposed = clamp(value * exp2(exposure), 0.0, 1.0);
        let square = exposed * exposed;
        let curve_delta = square * (3.0 - 2.0 * exposed) - exposed;
        let curved = clamp(contrast * curve_delta + exposed, 0.0, 1.0);
        let first_gamma = exp2(log2(max(curved, 1e-7)));
        color[c] = exp2(log2(max(first_gamma, 1e-7)));
    }
    let hsl = nr_hsl(color);
    let adjusted = nr_rgb(vec3(hsl.x, clamp(hsl.y * (saturation + 1.0), 0.0, 1.0), hsl.z));
    let again = nr_hsl(adjusted);
    return clamp(nr_rgb(vec3(again.x, clamp(exp2(log2(max(again.y, 1e-7))), 0.0, 1.0), again.z)), vec3(0.0), vec3(1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) { return; }
    let p = proxy[id.y * params.width + id.x].rgb;
    // The head is field-indexed (the padded field the network ran on).
    let h = head[id.y * params.full_width + id.x];
    var neural = clamp(p + h.rgb / 4.0, vec3(0.0), vec3(1.0));
    // Temporal blend with the reprojected previous output, where this pixel has a history.
    let valid = vec2(f32(params.width), f32(params.height));
    let mv = textureLoad(motion_tex, vec2<i32>(id.xy), 0).xy;
    let prev_uv = history_uv(vec2(f32(id.x), f32(id.y)), valid, mv, vec2(params.jitter_dx, params.jitter_dy));
    if (params.history_valid != 0u && on_screen(prev_uv)) {
        let hist = history_sample(prev_uv, valid, history_tex, lin_sampler);
        let weight = clamp(sigmoid(h.a) * clamp(params.blend_scale, 0.0, 1.0), 0.0, 1.0);
        neural = mix(neural, hist, weight);
    }
    // Publish the history for the next frame: truncated to the half grid, before the grade.
    textureStore(history_next, vec2<i32>(id.xy), vec4(truncate3(neural), 1.0));
    if (params.style != 0.0) {
        neural = nr_style(neural, params.style, params.local_tone);
    }
    neural = clamp(params.intensity * (neural - p) + p, vec3(0.0), vec3(1.0));
    var v = srgb_decode(neural);
    let over = v > vec3(0.75);
    let inv = vec3(0.75) - log(max(vec3(1.0) - (v - vec3(0.75)) / 0.25, vec3(1e-6))) / 5.770780;
    v = select(v, inv, over);
    let hdr = v * max(params.paper_white, 0.05) / max(params.exposure, 1e-6);
    let alpha = textureLoad(out_color, vec2<i32>(id.xy)).a;
    textureStore(out_color, vec2<i32>(id.xy), vec4(hdr, alpha));
}
"#;

/// The composite shader with the shared helpers prepended.
pub fn composite_source() -> String {
    format!("{ROUND_F16}\n{TEMPORAL}\n{COMPOSITE}")
}

/// The encode shader with the half-rounding helper prepended.
pub fn encode_source() -> String {
    format!("{ROUND_F16}\n{ENCODE}")
}

/// Temporal-path helpers shared by the preprocess and the composite: truncation to the half
/// grid, the noise sequence, and the five-tap Catmull-Rom history reconstruction (ported from
/// the OpenDLSS-NR demo's MIT nr_common.glsl / nr_preprocess.comp).
const TEMPORAL: &str = r#"
fn truncate_half(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let sign_bit = (bits >> 16u) & 0x8000u;
    let exponent = (bits >> 23u) & 0xffu;
    let mantissa = bits & 0x7fffffu;
    var half_bits: u32;
    if (exponent == 0xffu) {
        half_bits = sign_bit | select(0x7c00u, 0x7e00u, mantissa != 0u);
    } else {
        let half_exponent = i32(exponent) - 112;
        if (half_exponent >= 31) {
            half_bits = sign_bit | 0x7c00u;
        } else if (half_exponent <= 0) {
            if (half_exponent < -10) {
                half_bits = sign_bit;
            } else {
                half_bits = sign_bit | ((mantissa | 0x800000u) >> u32(14 - half_exponent));
            }
        } else {
            half_bits = sign_bit | (u32(half_exponent) << 10u) | (mantissa >> 13u);
        }
    }
    return unpack2x16float(half_bits).x;
}
fn truncate3(v: vec3<f32>) -> vec3<f32> {
    return vec3(truncate_half(v.x), truncate_half(v.y), truncate_half(v.z));
}
fn round3(v: vec3<f32>) -> vec3<f32> {
    return vec3(round_f16(v.x), round_f16(v.y), round_f16(v.z));
}
fn hash_uniform(value: u32) -> f32 {
    var mixed = value;
    mixed = (mixed >> ((mixed >> 28u) + 4u)) ^ mixed;
    mixed *= 0x108ef2d9u;
    let integer = ((mixed >> 30u) ^ (mixed >> 8u)) + 1u;
    return f32(integer) * bitcast<f32>(0x33800000u);
}
fn gaussian3(x: u32, y: u32, seed: u32) -> vec3<f32> {
    var base = (x * 0x8da6b343u) ^ (y * 0xd8163841u) ^ (seed * 0x9e3779b9u) ^ 0x243f6a88u;
    base = (base >> ((base >> 28u) + 4u)) ^ base;
    base *= 0x108ef2d9u;
    base = (base >> 22u) ^ base;
    let u0 = hash_uniform(base * 0x2c9277b5u + 0xac564b05u);
    let u1 = hash_uniform(base * 0xfa6dc5f9u + 0x4712a88eu);
    let u2 = hash_uniform(base * 0xcaa5b80du + 0x21dd796bu);
    let u3 = hash_uniform(base * 0x83232c31u + 0x3463e0acu);
    let radius0 = sqrt(log2(u0) * bitcast<f32>(0x3f317218u) * -2.0);
    let radius1 = sqrt(log2(u2) * bitcast<f32>(0x3f317218u) * -2.0);
    let tau = bitcast<f32>(0x40c90fdbu);
    return vec3(round_f16(radius0 * cos(u1 * tau)), round_f16(radius0 * sin(u1 * tau)), round_f16(radius1 * cos(u3 * tau)));
}
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp2(-x * 1.4426950408889634));
}
// Five-tap Catmull-Rom (four axis taps plus centre, bilinear-weight trick) of the history at
// `prev_uv`; a box or bilinear filter here softens the picture frame over frame.
fn history_sample(prev_uv: vec2<f32>, valid: vec2<f32>, tex: texture_2d<f32>, samp: sampler) -> vec3<f32> {
    let position = prev_uv * valid;
    let base = floor(position - 0.5) + 0.5;
    let f = clamp(position - base, vec2(0.0), vec2(1.0));
    let square = f * f;
    let cube = f * square;
    let w0 = (f + cube) * -0.5 + square;
    let w1 = (cube * 1.5 - square * 2.5) + 1.0;
    let w3 = (cube - square) * 0.5;
    let w2 = ((1.0 - w0) - w1) - w3;
    let middle = w1 + w2;
    let low = clamp(base - 1.0, vec2(0.5), valid - 0.5) / valid;
    let center = clamp(base + w2 / middle, vec2(0.5), valid - 0.5) / valid;
    let high = clamp(base + 2.0, vec2(0.5), valid - 0.5) / valid;
    let left = textureSampleLevel(tex, samp, vec2(low.x, center.y), 0.0).rgb;
    let top = textureSampleLevel(tex, samp, vec2(center.x, low.y), 0.0).rgb;
    let mid = textureSampleLevel(tex, samp, center, 0.0).rgb;
    let bottom = textureSampleLevel(tex, samp, vec2(center.x, high.y), 0.0).rgb;
    let right = textureSampleLevel(tex, samp, vec2(high.x, center.y), 0.0).rgb;
    let a = w0.x * middle.y;
    let b = w0.y * middle.x;
    let c = middle.x * middle.y;
    let d = w3.y * middle.x;
    let e = w3.x * middle.y;
    var value = left * a + top * b;
    value = mid * c + value;
    value = bottom * d + value;
    value = right * e + value;
    return value * (1.0 / (e + (d + (c + (a + b)))));
}
// Where this pixel was last frame, in uv: engine MVs (current-minus-previous, unjittered) plus
// the jitter delta between the two frames' offsets.
fn history_uv(pixel: vec2<f32>, valid: vec2<f32>, mv: vec2<f32>, jitter: vec2<f32>) -> vec2<f32> {
    return (pixel + 0.5) / valid + mv + jitter;
}
fn on_screen(uv: vec2<f32>) -> bool {
    return all(uv >= vec2(0.0)) && all(uv < vec2(1.0));
}
"#;

pub const PREPROCESS: &str = r#"
struct Params {
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
@group(0) @binding(0) var<storage, read> proxy: array<vec4<f32>>;
@group(0) @binding(1) var motion_tex: texture_2d<f32>;
@group(0) @binding(2) var history_tex: texture_2d<f32>;
@group(0) @binding(3) var lin_sampler: sampler;
@group(0) @binding(4) var<storage, read_write> features: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.full_width || id.y >= p.full_height) { return; }
    let rx = select(2 * i32(p.valid_width) - i32(id.x) - 2, i32(id.x), id.x < p.valid_width);
    let ry = select(2 * i32(p.valid_height) - i32(id.y) - 2, i32(id.y), id.y < p.valid_height);
    let sx = u32(clamp(rx, 0, i32(p.valid_width) - 1));
    let sy = u32(clamp(ry, 0, i32(p.valid_height) - 1));
    let centered = round3((round3(proxy[sy * p.valid_width + sx].rgb) - 0.5) * 0.125);
    let noise = gaussian3(id.x, id.y, p.seed);
    let valid = vec2(f32(p.valid_width), f32(p.valid_height));
    let pixel = vec2(f32(sx), f32(sy));
    let mv = textureLoad(motion_tex, vec2<i32>(i32(sx), i32(sy)), 0).xy;
    let prev_uv = history_uv(pixel, valid, mv, vec2(p.jitter_dx, p.jitter_dy));
    var history = centered;
    if (p.history_valid != 0u && on_screen(prev_uv)) {
        let h = history_sample(prev_uv, valid, history_tex, lin_sampler);
        history = round3((round3(h) - 0.5) * 0.125);
    }
    let base = (id.y * p.full_width + id.x) * 4u;
    features[base + 0u] = vec4(noise, 1.0);
    features[base + 1u] = vec4(centered, history.r);
    features[base + 2u] = vec4(history.g, history.b, p.style / 128.0, round_f16(p.tone));
    features[base + 3u] = vec4(
        round_f16(select(p.structure, 1.0, p.automask > 0.0)),
        round_f16(select(-1.0, select(p.structure, p.skin, p.skin < 0.0), p.automask > 0.0)),
        round_f16(select(-1.0, p.structure, p.automask > 0.0)),
        0.0,
    );
}
"#;

/// The preprocess shader with the shared helpers prepended.
pub fn preprocess_source() -> String {
    format!("{ROUND_F16}\n{TEMPORAL}\n{PREPROCESS}")
}

