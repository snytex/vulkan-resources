#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(push_constant) uniform PushConstants
{
  mat4 viewProj;
  vec4 camPosFog; // xyz = cam pos, w = fog dens
} pc;

layout(location = 0) out vec3 vColor;
layout(location = 1) out float vDistance;

void main()
{
  gl_Position = pc.viewProj * vec4(inPosition, 1.0);
  vColor = inColor;
  vDistance = length(inPosition - pc.camPosFog.xyz);
}
