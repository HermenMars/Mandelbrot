#version 450

layout(location = 0) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants
{
    vec2 center;
    float scale;
    float aspect;
    int maxIterations;
} pc;

vec3 palette(float t)
{
    return 0.5 + 0.5 * cos(
        6.28318 * (vec3(0.0, 0.33, 0.67) + t)
    );
}

void main()
{
    float x = (fragUV.x * 2.0 - 1.0)
            * pc.aspect * pc.scale;

    float y = (1.0 - fragUV.y * 2.0)
            * pc.scale;

    vec2 c = pc.center + vec2(x, y);

    vec2 z = vec2(0.0);

    int iterations = 0;

    float magnitude = 0.0;

    for (int i = 0; i < pc.maxIterations; i++)
    {
        float zx = z.x * z.x - z.y * z.y + c.x;
        float zy = 2.0 * z.x * z.y + c.y;

        z = vec2(zx, zy);

        magnitude = dot(z, z);

        if (magnitude > 256.0)
            break;

        iterations++;
    }

    if (iterations >= pc.maxIterations)
    {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float smoothIteration = float(iterations)
        + 1.0
        - log2(max(0.000001, log2(sqrt(magnitude))));

    float t = smoothIteration * 0.025;

    vec3 color = palette(t);

    outColor = vec4(color, 1.0);
}