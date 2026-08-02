#version 450

layout(location = 0) in vec2 vNdc;

layout(push_constant) uniform PushConstants
{
  mat4 viewProj;
  vec4 camPosFog; // xyz = cam pos,  w = fog density
  vec4 camRightTan; // xyz = right,    w = tan(fovY * 0.5)
  vec4 camUpAspect; // xyz = up,       w = aspect
  vec4 camFwdTime; // xyz = forward,  w = elapsed seconds
} pc;

layout(location = 0) out vec4 outColor;

const vec3 kFogColor = vec3(0.020, 0.022, 0.035);

float hash13(vec3 p)
{
  p = fract(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 p)
{
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);

  float n000 = hash13(i + vec3(0, 0, 0));
  float n100 = hash13(i + vec3(1, 0, 0));
  float n010 = hash13(i + vec3(0, 1, 0));
  float n110 = hash13(i + vec3(1, 1, 0));
  float n001 = hash13(i + vec3(0, 0, 1));
  float n101 = hash13(i + vec3(1, 0, 1));
  float n011 = hash13(i + vec3(0, 1, 1));
  float n111 = hash13(i + vec3(1, 1, 1));

  return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
    mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

float fbm(vec3 p)
{
  float sum = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 4; ++i)
  {
    sum += amp * vnoise(p);
    p = p * 2.03 + vec3(11.3, 7.7, 3.1);
    amp *= 0.5;
  }
  return sum;
}

// one layer of jittered point stars on a cubic lattice cut by the view sphere
float starLayer(vec3 d, float scale, float threshold, float t)
{
  vec3 p = d * scale;
  vec3 cell = floor(p);
  vec3 f = p - cell - 0.5;

  float h = hash13(cell);
  if (h < threshold) return 0.0;

  vec3 jitter = vec3(hash13(cell + 17.0), hash13(cell + 41.0), hash13(cell + 73.0)) - 0.5;
  float dist = length(f - jitter * 0.7);

  float bright = smoothstep(threshold, 1.0, h);
  float twinkle = 0.65 + 0.35 * sin(t * 2.3 + h * 97.0);
  return bright * twinkle * exp(-dist * dist * 110.0);
}

void main()
{
  float tanHalf = pc.camRightTan.w;
  float aspect = pc.camUpAspect.w;
  float t = pc.camFwdTime.w;

  // vulkan ndc has +y down, so the up axis is negated here
  vec3 dir = normalize(pc.camFwdTime.xyz
        + pc.camRightTan.xyz * (vNdc.x * tanHalf * aspect)
        - pc.camUpAspect.xyz * (vNdc.y * tanHalf));

  // base void gradient: slightly warmer toward the horizon
  vec3 col = mix(vec3(0.014, 0.016, 0.032), vec3(0.003, 0.004, 0.011),
      smoothstep(-0.15, 0.75, dir.y));

  // nebula: domain-warped fbm, drifting very slowly
  vec3 q = dir * 2.4 + vec3(0.0, 0.0, t * 0.006);
  float warp = fbm(q);
  float n = fbm(q * 1.9 + warp * 1.6 + vec3(3.7, 1.2, -2.4));

  float cloud = smoothstep(0.36, 0.88, warp * 0.55 + n * 0.65);
  vec3 cool = vec3(0.06, 0.26, 0.55);
  vec3 warm = vec3(0.42, 0.09, 0.52);
  col += mix(cool, warm, smoothstep(0.25, 0.85, n)) * cloud * 0.95;

  col += vec3(0.95, 0.55, 0.30) * pow(cloud, 5.0) * 0.6;

  float lane = smoothstep(0.55, 0.30, fbm(dir * 6.1 + 19.0));
  col *= mix(1.0, 0.55, lane * cloud);

  float s = starLayer(dir, 140.0, 0.982, t) * 1.6
      + starLayer(dir, 300.0, 0.975, t) * 0.8
      + starLayer(dir, 620.0, 0.968, t) * 0.35;
  col += vec3(0.85, 0.90, 1.0) * s;

  const vec3 kSunDir = normalize(vec3(0.82, 0.26, 0.51));
  float sd = max(dot(dir, kSunDir), 0.0);
  col += vec3(0.55, 0.72, 1.0) * pow(sd, 900.0) * 6.0;
  col += vec3(0.20, 0.34, 0.62) * pow(sd, 24.0) * 0.55;

  col = mix(kFogColor, col, smoothstep(-0.03, 0.28, dir.y));

  outColor = vec4(col, 1.0);
}
