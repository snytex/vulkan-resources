#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(push_constant) uniform PushConstants
{
  mat4 viewProj;
  vec4 camPosFog;   // xyz = cam pos, w = fog dens
  vec4 camRightTan; // xyz = right,   w = tan(fovY * 0.5)
  vec4 camUpAspect; // xyz = up,      w = aspect
  vec4 camFwdTime;  // xyz = forward, w = elapsed seconds
} pc;

layout(location = 0) out vec3 vColor;
layout(location = 1) out float vDistance;

void main()
{
  gl_Position = pc.viewProj * vec4(inPosition, 1.0);
  vColor = inColor;
  vDistance = length(inPosition - pc.camPosFog.xyz);
}
