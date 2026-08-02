#version 450

// Fully procedural: nothing is bound, everything comes from gl_InstanceIndex.
// 6 vertices per instance -> one camera-facing billboard quad.
//
// Instances [0, kGalaxyCount)      -> distant spiral galaxy
// Instances [kGalaxyCount, total)  -> dust motes wrapped around the camera
//
// The total instance count lives in main.cpp (kParticleCount); keep them in sync.
const uint kGalaxyCount = 60000u;

layout(push_constant) uniform PushConstants
{
  mat4 viewProj;
  vec4 camPosFog;   // xyz = cam pos,  w = fog density
  vec4 camRightTan; // xyz = right,    w = tan(fovY * 0.5)
  vec4 camUpAspect; // xyz = up,       w = aspect
  vec4 camFwdTime;  // xyz = forward,  w = elapsed seconds
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vColor;
layout(location = 2) out float vIntensity;

const vec2 kCorners[6] = vec2[6]
(
  vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
  vec2(-1.0, -1.0), vec2(1.0,  1.0), vec2(-1.0, 1.0)
);

const float kTau = 6.28318530718;

const vec3  kGalaxyCenter = vec3(0.0, 110.0, -300.0);
const float kGalaxyRadius = 150.0;
const float kGalaxyTilt   = 0.96; // radians, ~55 degrees off face-on
const float kDustRange    = 70.0;

float hash11(float n)
{
  return fract(sin(n * 12.9898) * 43758.5453123);
}

void galaxyParticle(float fid, float t, out vec3 pos, out vec3 color, out float size, out float intensity)
{
  float a0 = hash11(fid * 1.13 + 0.7);
  float a1 = hash11(fid * 2.37 + 5.1);
  float a2 = hash11(fid * 3.71 + 13.3);
  float a3 = hash11(fid * 5.19 + 29.9);

  const float kArms = 4.0;
  float arm = floor(a0 * kArms);

  // denser toward the core
  float radius = pow(a1, 0.55) * kGalaxyRadius + 5.0;

  // arm spread widens outward; a few strays scatter into the halo
  float spread = (a2 - 0.5) * (0.30 + radius * 0.0035);
  float halo = step(0.965, a3) * (a2 - 0.5) * 3.0;

  // differential rotation: the core winds faster than the rim
  float omega = 6.0 / (radius * 0.35 + 30.0);
  float theta = arm * (kTau / kArms) + radius * 0.021 + spread + halo + t * omega;

  // disk basis, tilted about X so the spiral reads open from the ground
  vec3 u = vec3(1.0, 0.0, 0.0);
  vec3 n = vec3(0.0, cos(kGalaxyTilt), sin(kGalaxyTilt));
  vec3 v = vec3(0.0, sin(kGalaxyTilt), -cos(kGalaxyTilt));

  float thickness = (hash11(fid * 7.31 + 3.3) - 0.5) * (26.0 * exp(-radius * 0.016) + 3.0);

  pos = kGalaxyCenter + u * (cos(theta) * radius) + v * (sin(theta) * radius) + n * thickness;

  // core: white-hot -> gold -> blue disk -> magenta HII knots at the rim
  float rn = radius / kGalaxyRadius;
  vec3 core = vec3(1.00, 0.94, 0.78);
  vec3 mid  = vec3(0.55, 0.72, 1.00);
  vec3 rim  = vec3(0.85, 0.35, 0.75);
  color = mix(mix(core, mid, smoothstep(0.04, 0.42, rn)), rim, smoothstep(0.55, 1.0, rn));

  // a scattering of bright young stars punched through the arms
  float bright = step(0.993, hash11(fid * 11.7 + 61.0));
  color = mix(color, vec3(0.75, 0.90, 1.0), bright);

  size = mix(1.1, 2.6, a2) + bright * 3.0;
  intensity = (0.14 + 0.55 * exp(-rn * 2.2)) + bright * 0.9;
}

void dustParticle(float fid, float t, out vec3 pos, out vec3 color, out float size, out float intensity)
{
  vec3 base = vec3(hash11(fid * 1.7), hash11(fid * 3.1 + 9.0), hash11(fid * 4.9 + 21.0)) * kDustRange;
  vec3 vel = vec3(hash11(fid * 6.3 + 2.0) - 0.5, hash11(fid * 8.1 + 4.0) * 0.5 + 0.15,
                  hash11(fid * 9.7 + 6.0) - 0.5) * 1.6;

  // keep the field centred on the camera by wrapping into a moving cube
  vec3 p = base + vel * t - pc.camPosFog.xyz + kDustRange * 0.5;
  pos = mod(p, vec3(kDustRange)) - kDustRange * 0.5 + pc.camPosFog.xyz;

  float warm = step(0.90, hash11(fid * 13.3 + 77.0));
  color = mix(vec3(0.45, 0.72, 0.95), vec3(1.00, 0.62, 0.30), warm);

  float twinkle = 0.55 + 0.45 * sin(t * 1.7 + fid * 0.37);
  size = 0.05 + hash11(fid * 15.1) * 0.10 + warm * 0.06;
  intensity = (0.25 + 0.35 * warm) * twinkle;
}

void main()
{
  float fid = float(gl_InstanceIndex);
  float t = pc.camFwdTime.w;

  vec3 pos;
  vec3 color;
  float size;
  float intensity;

  if (uint(gl_InstanceIndex) < kGalaxyCount)
    galaxyParticle(fid, t, pos, color, size, intensity);
  else
    dustParticle(fid - float(kGalaxyCount), t, pos, color, size, intensity);

  // hold a minimum angular size so far sprites never fall below a pixel
  float dist = distance(pos, pc.camPosFog.xyz);
  size = max(size, dist * 0.0016);

  // fade sprites that drift into the lens, otherwise they bloom into blobs
  intensity *= smoothstep(0.0, 5.0, dist);

  vec2 corner = kCorners[gl_VertexIndex];
  vec3 world = pos + pc.camRightTan.xyz * (corner.x * size) + pc.camUpAspect.xyz * (corner.y * size);

  gl_Position = pc.viewProj * vec4(world, 1.0);
  vUV = corner;
  vColor = color;
  vIntensity = intensity;
}
