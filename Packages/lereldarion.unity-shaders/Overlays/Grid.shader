// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// An overlay which displays a 1m grid overlapped onto world, from data sampled in the depth texture. Requires dynamic lighting to work (for the depth texture).
//
// Initial idea from https://github.com/netri/Neitri-Unity-Shaders
// Improved with SPS-I support, Fullscreen "screenspace" mode.
// Rewritten way of recreating VS positions using interpolated VS ray : precise, removes inverse, avoids unavailable unity_MatrixInvP.

Shader "Lereldarion/Overlay/Grid" {
    Properties {
        _Grid_Zoom("Grid Zoom", Float) = 1
        [ToggleUI] _Overlay_Fullscreen("Force Screenspace Fullscreen", Float) = 0
    }
    SubShader {
        Tags {
            "Queue" = "Overlay"
            "RenderType" = "Overlay"
            "VRCFallback" = "Hidden"
            "PreviewType" = "Plane"
        }
        
        Cull Off
        Blend One One // Emission
        ZWrite Off
        ZTest Less

        Pass {
            CGPROGRAM
            #pragma target 5.0
            #pragma vertex vertex_stage
            #pragma geometry geometry_stage
            #pragma fragment fragment_stage
            #pragma multi_compile_instancing
            
            #include "UnityCG.cginc"

            struct VertexInput {
                float4 position_os : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct FragmentInput {
                float4 position : SV_POSITION; // CS as rasterizer input, screenspace as fragment input
                float3 position_vs : POSITION_VS;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            void vertex_stage (VertexInput input, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                output.position_vs = UnityObjectToViewPos(input.position_os);
                output.position = UnityViewToClipPos(output.position_vs);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
            }
            
            uniform float _Overlay_Fullscreen;
            uniform float _VRChatMirrorMode;
            uniform float _VRChatCameraMode;

            [maxvertexcount(4)]
            void geometry_stage(triangle FragmentInput input[3], uint triangle_id : SV_PrimitiveID, inout TriangleStream<FragmentInput> stream) {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input[0]);
                if(_Overlay_Fullscreen == 1 && _VRChatMirrorMode == 0 && _VRChatCameraMode == 0) {
                    // Fullscreen mode : generate a fullscreen quad for triangle 0 and discard others
                    if (triangle_id == 0) {
                        FragmentInput output = input[0];

                        // Generate in VS close to near clip plane. Having non CS positions is essential to return to WS later.
                        float2 quad[4] = { float2(-1, -1), float2(-1, 1), float2(1, -1), float2(1, 1) };
                        float near_plane_z = -_ProjectionParams.y;
                        float2 tan_half_fov = 1 / unity_CameraProjection._m00_m11; // https://jsantell.com/3d-projection/
                        // Add margins, mostly in case of oblique P matrices or similar
                        float quad_z = near_plane_z * 2; // z margin
                        float quad_xy = quad_z * tan_half_fov * 1.2; // xy margin

                        UNITY_UNROLL
                        for(uint i = 0; i < 4; i += 1) {
                            output.position_vs = float4(quad[i] * quad_xy, quad_z, 1);
                            output.position = UnityViewToClipPos(output.position_vs);
                            stream.Append(output);
                        }
                    }
                } else {
                    // Normal geometry mode : forward triangle
                    stream.Append(input[0]);
                    stream.Append(input[1]);
                    stream.Append(input[2]);
                }
            }

            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
            float4 _CameraDepthTexture_TexelSize;

            struct SceneReconstruction {
                float2 pixel;
                float3 ray_vs;
                float3 ray_dx_vs;
                float3 ray_dy_vs;

                static SceneReconstruction init(FragmentInput input) {
                    SceneReconstruction o;
                    o.pixel = input.position.xy;
                    o.ray_vs = input.position_vs / input.position.w;
                    // Use derivatives to get ray for neighbouring pixels.
                    // This is exact because ray is linear on a fragment.
                    // TODO improve by using ddx/ddy on separate position_vs / position.w and then divide ?
                    o.ray_dx_vs = ddx(o.ray_vs);
                    o.ray_dy_vs = ddy(o.ray_vs);
                    return o;
                }

                float3 position_vs() {
                    return position_vs(float2(0, 0));
                }
                
                float3 position_vs(float2 pixel_shift) {
                    // HLSLSupport.hlsl : DepthTexture is a TextureArray in SPS-I, so its size should be safe to use to get uvs.
                    float3 shifted_ray_vs = ray_vs + pixel_shift.x * ray_dx_vs + pixel_shift.y * ray_dy_vs;
                    float2 uv = (pixel + pixel_shift) * _CameraDepthTexture_TexelSize.xy;
                    float raw = SAMPLE_DEPTH_TEXTURE_LOD(_CameraDepthTexture, float4(uv, 0, 0)); // [0,1]
                    if (!(0 < raw && raw < 1)) { discard; }
                    return shifted_ray_vs * LinearEyeDepth(raw);
                }
            };

            float3 distance_to_01_grid_lines(float3 position) {
                return abs(frac(0.5 + position) - 0.5);
            }

            uniform float _Grid_Zoom;

            fixed4 fragment_stage (FragmentInput input) : SV_Target {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                SceneReconstruction sr = SceneReconstruction::init(input);
                float3 vs_0_0 = sr.position_vs();
                float3 ws = mul(unity_MatrixInvV, float4(vs_0_0, 1)).xyz;
                                
                // Idea : add tint if position is close to axis xyz at grid_size intervals;
                float3 grid_distance = distance_to_01_grid_lines(_Grid_Zoom * ws);
                // non-linearize to spike if distance is 0
                float3 grid_proximity = saturate(0.5 - grid_distance * grid_distance * 1000);
                // Already fits Blender colors : x=red, y=green, z=blue
                return float4(grid_proximity, 1);
            }
            ENDCG
        }
    }
}
