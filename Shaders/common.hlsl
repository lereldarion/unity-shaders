#ifndef _LERELDARION_UNITY_SHADERS_COMMON_HLSL
#define _LERELDARION_UNITY_SHADERS_COMMON_HLSL

static const float f32_nan = asfloat(uint(-1)); // 0xFFF...FFF should be a quiet NaN

float length_sq(float3 v) { return dot(v, v); }
float length_sq(float2 v) { return dot(v, v); }

float3x3 billboard_referential(float3 billboard_normal, float3 up) {
    billboard_normal = normalize(billboard_normal);
    const float3 billboard_x = normalize(cross(up, billboard_normal));
    const float3 billboard_y = cross(billboard_normal, billboard_x);
    return float3x3(billboard_x, billboard_y, billboard_normal);
}

float2 ray_sphere_intersect(float3 origin, float3 ray_normalized, float3 sphere_center, float sphere_radius_sq) {
    // https://iquilezles.org/articles/intersectors/
    const float3 oc = origin - sphere_center;
    const float b = dot(oc, ray_normalized);
    const float c = dot(oc, oc) - sphere_radius_sq;
    float h = b * b - c;
    if(h < 0.0) { return float2(-1.0, -1.0); }
    h = sqrt(h);
    return float2(-b - h, -b + h);
}

// VRChat globals
uniform float _VRChatMirrorMode;
uniform float _VRChatCameraMode;

// unity_MatrixInvP is not provided in BIRP. unity_CameraInvProjection is only the basic camera projection (no VR components).
// Using d4rkpl4y3r technique of patching unity_CameraInvProjection (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3)
// Use after instance id have been set ! UNITY_SETUP_INSTANCE_ID(input)
static float4x4 unity_birp_MatrixInvP;
static float4x4 unity_birp_MatrixInvMVP;
void setup_unity_birp_MatrixInvP() {
    float4x4 flipZ = float4x4(1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, -1, 1,
                            0, 0, 0, 1);
    float4x4 scaleZ = float4x4(1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 2, -1,
                            0, 0, 0, 1);
    float4x4 invP = unity_CameraInvProjection;
    float4x4 flipY = float4x4(1, 0, 0, 0,
                            0, _ProjectionParams.x, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1);
    float4x4 m = mul(scaleZ, flipZ);
    m = mul(invP, m);
    m = mul(flipY, m);
    m._24 *= _ProjectionParams.x;
    m._42 *= -1;
    unity_birp_MatrixInvP = m;
    unity_birp_MatrixInvMVP = mul(unity_WorldToObject, mul(unity_MatrixInvV, unity_birp_MatrixInvP));
}

///////////////////////////////////////////////////////////////////////////////
// Overlay mode common code.
// 
// The functions below handle the various overlay modes :
// - Mesh : simply apply on the input mesh
// - Fullscreen : apply the overlay as fullscreen, depending on vrchat camera config
// - Billboard sphere : impostor sphere ar object origin, using the mesh as a billboard. When inside the fake sphere go in fullscreen.
//
// These modes are selected by multi_compile macros.
// The logic is annoying due to the many static if branch, so it has been extracted here to keep the overlay-specific code clean :
// - add the required properties in the property block
// - add the multi_compile variants. Omit multi_compile for modes that are not to be defined.
// - insert the Overlay*Extra structs into their targets
// - call the overlay_vertex_clip_pos() in vertex and overlay_fragment() in fragment 

uniform uint _Overlay_Fullscreen_Vertex_Order;
uniform float _Overlay_Fullscreen_Only_Main_Camera;
uniform float _Overlay_Sphere_Filled;
uniform float2 _Overlay_Radial_Dissolve_Bounds;

UNITY_DECLARE_TEX2D(_Overlay_Radial_Dissolve_Noise_Texture);
uniform float4 _Overlay_Radial_Dissolve_Noise_Texture_ST;

struct OverlayFragmentInputExtra {
    // Overlay specific data for FragmentInput struct. Depend on static mode.

    #if defined(_OVERLAY_RADIAL_DISSOLVE_ENABLED)
    float2 disk_uv : DISK_UV;
    #endif

    #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
    float3 position_os : POSITION_OS;
    float3 ray_os : RAY_OS;
    float sphere_radius_sq_os : SPHERE_RADIUS_SQ_OS;
    #endif
};

struct OverlayFragmentOutputExtra {
    // Overlay specific data for FragmentOutput struct. Depend on static mode.

    #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
    // Place the billboard in front of the fake sphere, and set sphere depth further away. Use conservative depth to keep early Z culling.
    // https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#ConservativeoDepth
    #if defined(UNITY_REVERSED_Z)
    float depth : SV_DepthLessEqual; // far < near (depths), [0, 1] on DX11W
    #else
    float depth : SV_DepthGreaterEqual; // near > far
    #endif
    #endif
};

bool sphere_os_intersects_near_quad(float3 center, float radius_sq) {
    // Near quad corners (object space)
    float4 near_00 = mul(unity_birp_MatrixInvMVP, float4(-1, -1, UNITY_NEAR_CLIP_VALUE, 1)); near_00.xyz /= near_00.w;
    float4 near_10 = mul(unity_birp_MatrixInvMVP, float4( 1, -1, UNITY_NEAR_CLIP_VALUE, 1)); near_10.xyz /= near_00.w;
    float4 near_01 = mul(unity_birp_MatrixInvMVP, float4(-1,  1, UNITY_NEAR_CLIP_VALUE, 1)); near_01.xyz /= near_00.w;
    float4 near_11 = mul(unity_birp_MatrixInvMVP, float4( 1,  1, UNITY_NEAR_CLIP_VALUE, 1)); near_11.xyz /= near_00.w;
    // Too annoying to compute precise distance to quad. Approximate with a disk
    const float3 quad_center = (near_00.xyz + near_01.xyz + near_10.xyz + near_11.xyz) / 4.0;
    const float quad_radius_sq = max(max(length_sq(near_00.xyz - quad_center), length_sq(near_10.xyz - quad_center)), max(length_sq(near_01.xyz - quad_center), length_sq(near_11.xyz - quad_center)));

    const float3 near_quad_normal = cross(near_00.xyz - quad_center, near_10.xyz - quad_center);
    const float3 projected_sphere_center = center + (dot(quad_center - center, near_quad_normal) / length_sq(near_quad_normal)) * near_quad_normal;
    const float projected_sphere_radius_sq = radius_sq - length_sq(projected_sphere_center - center);
    if(projected_sphere_radius_sq <= 0) { return false; } // Sphere too far from plane

    // Disk intersection : squared version of : length(projected_sphere_center - quad_center) <= projected_sphere_radius + quad_radius
    return length_sq(projected_sphere_center - quad_center) <= projected_sphere_radius_sq + 2 * sqrt(projected_sphere_radius_sq * quad_radius_sq) + quad_radius_sq;
}

bool overlay_radial_dissolve_should_discard(float2 disk_uv) {
    // Radial dissolve pattern : returns true if discard is needed.

    const float radius = length(disk_uv);
    const float transition = (radius - _Overlay_Radial_Dissolve_Bounds.x) / (_Overlay_Radial_Dissolve_Bounds.y - _Overlay_Radial_Dissolve_Bounds.x);
    const float transition_clamped = saturate(transition);
    if(transition != transition_clamped) { return transition > 0; } // Fully kept or discarded, do not sample noise.
    const float threshold = transition_clamped * transition_clamped; // [0, 1] -> [0, 1], starts slow but fast end.

    const float2 polar = _Overlay_Radial_Dissolve_Noise_Texture_ST.xy * float2(atan2(disk_uv.y, disk_uv.x) / UNITY_TWO_PI, radius) +  _Time.y * _Overlay_Radial_Dissolve_Noise_Texture_ST.zw;
    const float2 noise_scales = float2(1, 0.5);
    const float noise = dot(noise_scales / dot(1, noise_scales), float2(
        UNITY_SAMPLE_TEX2D_LOD(_Overlay_Radial_Dissolve_Noise_Texture, polar, 0).r,
        UNITY_SAMPLE_TEX2D_LOD(_Overlay_Radial_Dissolve_Noise_Texture, polar * float2(-2, 2), 0).r
    ));
    return noise < threshold;
}

float4 OverlayObjectToClipPos(float3 position_os, float2 uv0, uint vertex_id, out OverlayFragmentInputExtra output) {
    // Computes the clip space position, and extra data, depending on overlay mode.
    float4 position_cs;

    float2 disk_uv = 2 * uv0 - 1;

    // Determine if we need fullscreen fragment
    #if defined(_OVERLAY_MODE_MESH)
    const bool fullscreen = false;
    #elif defined(_OVERLAY_MODE_FULLSCREEN)
    const bool fullscreen = _VRChatMirrorMode == 0 && _VRChatCameraMode * _Overlay_Fullscreen_Only_Main_Camera == 0;
    #elif defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
    const float sphere_radius_sq_os = length_sq(position_os) / max(length_sq(disk_uv), 1e-6);
    const float sphere_radius_os = sqrt(sphere_radius_sq_os);
    output.sphere_radius_sq_os = sphere_radius_sq_os;

    const bool is_orthographic = UNITY_MATRIX_P._m33 == 1.0;
    const float3 camera_forward_os = mul(unity_WorldToObject, mul(unity_MatrixInvV, float3(0, 0, -1)));
    const float3 camera_pos_os = mul(unity_WorldToObject, float4(_WorldSpaceCameraPos, 1)).xyz;

    const bool fullscreen = sphere_os_intersects_near_quad(float3(0, 0, 0), sphere_radius_sq_os);
    #endif

    if(fullscreen) {
        // Fullscreen mode : cover the screen with a quad by redirecting existing vertices
        if(vertex_id < 4) {
            const uint2 bits = vertex_id & uint2(2, 1); // [00, 01, 10, 11]
            const float2 ndc = bits ? 1 : -1; // [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            const float2 swap = _Overlay_Fullscreen_Vertex_Order & bits.yx ? -1 : 1;
            position_cs = float4(ndc * swap, UNITY_NEAR_CLIP_VALUE, 1);
            
            #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
            float4 v = mul(unity_birp_MatrixInvMVP, position_cs);
            output.position_os = v.xyz / v.w;
            disk_uv = 0; // Fullscreen inside sphere
            #else
            disk_uv = position_cs.xy * rsqrt(2); // Fullscreen radial dissolve
            #endif
        } else {
            position_cs = f32_nan.xxxx; // Vertex discard

            #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
            output.position_os = 0;
            #endif
        }
    } else {
        #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
        float apparent_radius_factor;
        float3 billboard_normal_os;

        // Place surface in front of the sphere (to use conservative depth) and shrunk to apparent size (perspective)
        if(is_orthographic) {
            billboard_normal_os = -camera_forward_os;
            apparent_radius_factor = 1.;
        } else {
            const float camera_distance_os = length(camera_pos_os);
            apparent_radius_factor = sqrt(max(camera_distance_os - sphere_radius_os, 0) / (camera_distance_os + sphere_radius_os));
            billboard_normal_os = camera_pos_os;
        }
        const float3 world_up_os = mul(unity_WorldToObject, float3(0, 1, 0));
        position_os = mul(sphere_radius_os * float3(disk_uv * apparent_radius_factor, 1), billboard_referential(billboard_normal_os, world_up_os));
        output.position_os = position_os;
        #endif

        position_cs = UnityObjectToClipPos(position_os);
    }

    #if defined(_OVERLAY_RADIAL_DISSOLVE_ENABLED)
    output.disk_uv = disk_uv;
    #endif
    #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
    output.ray_os = is_orthographic ? camera_forward_os : output.position_os - camera_pos_os;
    #endif

    return position_cs;
}

void OverlayFragment(OverlayFragmentInputExtra input, out OverlayFragmentOutputExtra output) {
    // In fragment, discard and patch depth, depending on overlay mode
    
    #if defined(_OVERLAY_RADIAL_DISSOLVE_ENABLED)
    if (overlay_radial_dissolve_should_discard(input.disk_uv)) { discard; }
    #endif
    
    #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)                
    const float3 ray_os = normalize(input.ray_os);
    const float2 ray_hits = ray_sphere_intersect(input.position_os, ray_os, float3(0, 0, 0), input.sphere_radius_sq_os);
    if(ray_hits.y < 0) {
        output.depth = 0; discard; // Outside and no intersect
    } else if(ray_hits.x < 0 && _Overlay_Sphere_Filled) {
        output.depth = UNITY_NEAR_CLIP_VALUE; // Inside sphere -> Fullscreen
    } else {
        // Display on first sphere intersection, compute proper depth
        const float4 sphere = UnityObjectToClipPos(input.position_os + (ray_hits.x >= 0 ? ray_hits.x : ray_hits.y) * ray_os);
        output.depth = sphere.z / sphere.w;
    }
    #endif
}

#endif