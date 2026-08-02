#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vColor;
layout(location = 2) in float vIntensity;

layout(location = 0) out vec4 outColor;

void main()
{
  float d2 = dot(vUV, vUV);
  if (d2 > 1.0) discard;

  // gaussian core that reaches exactly zero at the sprite edge
  float falloff = exp(-d2 * 4.0) - exp(-4.0);
  float glow = falloff * vIntensity;

  // small white-hot centre on top of the tinted halo
  vec3 col = vColor * glow + vec3(1.0) * pow(falloff, 6.0) * vIntensity * 0.5;

  outColor = vec4(col, 1.0); // additive blend, alpha unused
}
