#version 450

layout(location = 0) in vec3 vColor;
layout(location = 1) in float vDistance;

layout(push_constant) uniform PushConstants
{
  mat4 viewProj;
  vec4 camPosFog;
  vec4 camRightTan;
  vec4 camUpAspect;
  vec4 camFwdTime;
} pc;

layout(location = 0) out vec4 outColor;

const vec3 kFogColor = vec3(0.020, 0.022, 0.035);

void main()
{
  float fog = exp(-vDistance * pc.camPosFog.w);
  fog = clamp(fog, 0.0, 1.0);
  outColor = vec4(mix(kFogColor, vColor, fog), 1.0);
}
