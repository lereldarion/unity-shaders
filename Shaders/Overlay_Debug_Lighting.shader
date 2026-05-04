// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// Variant of the Debug/Lighting shader, with gizmos only visible through the overlay window.
// See debug_lighting.hlsl for gizmo documentation.
//
// Uses Stencil to identify the overlay window.
// Only the 1 bits are used and modified, so it can be compatible with other effect if you use different bits.

Shader "Lereldarion/Overlay/Debug Lighting" {
    Properties {  
        [KeywordEnum(Mesh, Fullscreen, Billboard Sphere)] _Overlay_Mode("Overlay mode", Float) = 0
        [IntRange] _Overlay_Fullscreen_Vertex_Order("Fullscreen vertex order (mesh dependent)", Range(0, 2)) = 0
        [ToggleUI] _Overlay_Fullscreen_Enable("Fullscreen mode : dynamic toggle", Float) = 1
        [ToggleUI] _Overlay_Fullscreen_Only_Main_Camera("Fullscreen mode : restricted to main camera", Float) = 1
        [Enum(Surface Only, 0, Filled, 1)] _Overlay_Sphere_Filled("Sphere type", Float) = 1
        [IntRange] _Overlay_Stencil("Stencil mask bits", Range(1, 255)) = 128 // 0b1000_0000
        
        [Header(Anchor)]
        [Enum(Mesh origin, 0, Camera, 1)] _Anchor_Mode ("Anchor for lightprobe sampler and directional lights", Float) = 0
        _Anchor_ViewSpace_Displacement ("Viewspace displacement from anchor position", Vector) = (0, 0, 0, 0)
        
        [Header(Gizmos sizes)]
        _Directional_Light_Arrow_Length ("Directional light arrow length", Float) = 0.5
        _LightProbe_Radius ("Light Probe sphere radius", Float) = 0.05
        _ReflectionProbe_Radius ("Reflection Probe sphere radius", Float) = 0.2
        _Light_Radius ("Radius of light center gizmos (sphere cookie, octahedron, ...)", Float) = 0.2
        _Minimum_Gizmo_To_Range_Ratio ("Reduce gizmo radius of short-ranged lights to keep this size ratio", Range(0, 1)) = 0.5
    }
    SubShader {
        Tags {
            "Queue" = "Overlay"
            "RenderType" = "Overlay"
            "VRCFallback" = "Hidden"
            "PreviewType" = "Plane"
            "IgnoreProjector" = "True"
            "DisableBatching" = "True" // For Billboard sphere mode only
        }

        // _Distance_Limit is undefined, defaults to 0, and disables the distance limit.

        Pass {
            Name "Set Stencil"

            ZTest LEqual
            ZWrite Off
            Stencil {
                Ref [_Overlay_Stencil]
                WriteMask [_Overlay_Stencil]
                Pass Replace
            }
            ColorMask 0 // Just set stencil
            Cull Off

            CGPROGRAM
            #pragma warning (error : 3205) // implicit precision loss
            #pragma warning (error : 3206) // implicit truncation

            #pragma target 5.0
            #pragma multi_compile_instancing
            #pragma shader_feature_local _OVERLAY_MODE_MESH _OVERLAY_MODE_FULLSCREEN _OVERLAY_MODE_BILLBOARD_SPHERE
            #pragma instancing_options procedural:vertInstancingSetup

            #pragma vertex vertex_stage
            #pragma fragment fragment_stage
            
            #include "UnityCG.cginc"
            #include "UnityStandardParticleInstancing.cginc"
            #include "common.hlsl"

            struct VertexInput {
                float3 position_os : POSITION;
                float2 uv0 : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct FragmentInput {
                float4 position : SV_POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
                OverlayFragmentInputExtra overlay_extra;
            };
            struct FragmentOutput {
                half4 color : SV_Target;
                OverlayFragmentOutputExtra overlay_extra;
            };
            
            void vertex_stage (VertexInput input, uint vertex_id : SV_VertexID, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                setup_unity_birp_MatrixInvP();
                output.position = OverlayObjectToClipPos(input.position_os, input.uv0, vertex_id, output.overlay_extra);
            }

            void fragment_stage (FragmentInput input, out FragmentOutput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                setup_unity_birp_MatrixInvP();
                OverlayFragment(input.overlay_extra, output.overlay_extra);
                output.color = half4(0, 0, 0, 0);
            }
            ENDCG
        }
        Pass {
            Name "Base Visible Lines"
            Tags { "LightMode" = "ForwardBase" }

            ZTest LEqual
            Stencil {
                Ref [_Overlay_Stencil]
                ReadMask [_Overlay_Stencil]
                Comp Equal
            }

            CGPROGRAM
            #include "debug_lighting.hlsl"
            #pragma vertex vertex_stage
            #pragma geometry geometry_lines_base
            #pragma fragment fragment_lines_visible
            #pragma multi_compile_instancing
            ENDCG
        }
        Pass {
            Name "Additive Visible Lines"
            Tags { "LightMode" = "ForwardAdd" }

            ZTest LEqual
            Stencil {
                Ref [_Overlay_Stencil]
                ReadMask [_Overlay_Stencil]
                Comp Equal
            }

            CGPROGRAM
            #include "debug_lighting.hlsl"
            #pragma vertex vertex_stage
            #pragma geometry geometry_lines_add
            #pragma fragment fragment_lines_visible
            #pragma multi_compile_instancing
            #pragma multi_compile_fwdadd
            ENDCG
        }
        Pass {
            Name "Base Spheres"
            Tags { "LightMode" = "ForwardBase" }

            ZTest LEqual
            Stencil {
                Ref [_Overlay_Stencil]
                ReadMask [_Overlay_Stencil]
                Comp Equal
            }

            CGPROGRAM
            #include "debug_lighting.hlsl"
            #pragma vertex vertex_stage
            #pragma geometry geometry_triangles_base
            #pragma fragment fragment_triangles
            #pragma multi_compile_instancing
            ENDCG
        }
        Pass {
            Name "Base Occluded Lines"
            Tags { "LightMode" = "ForwardBase" }

            ZTest Greater
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            Stencil {
                Ref [_Overlay_Stencil]
                ReadMask [_Overlay_Stencil]
                Comp Equal
            }

            CGPROGRAM
            #include "debug_lighting.hlsl"
            #pragma vertex vertex_stage
            #pragma geometry geometry_lines_base
            #pragma fragment fragment_lines_occluded
            #pragma multi_compile_instancing
            ENDCG
        }
        Pass {
            Name "Additive Occluded Lines"
            Tags { "LightMode" = "ForwardAdd" }

            ZTest Greater
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            Stencil {
                Ref [_Overlay_Stencil]
                ReadMask [_Overlay_Stencil]
                Comp Equal
            }

            CGPROGRAM
            #include "debug_lighting.hlsl"
            #pragma vertex vertex_stage
            #pragma geometry geometry_lines_add
            #pragma fragment fragment_lines_occluded
            #pragma multi_compile_instancing
            #pragma multi_compile_fwdadd
            ENDCG
        }
        Pass {
            Name "Clear Stencil"

            ZTest Always
            ZWrite Off
            Stencil {
                ReadMask [_Overlay_Stencil]
                Comp Equal
                WriteMask [_Overlay_Stencil]
                Pass Zero
            }
            ColorMask 0 // Just set stencil
            Cull Off

            CGPROGRAM
            #pragma warning (error : 3205) // implicit precision loss
            #pragma warning (error : 3206) // implicit truncation

            #pragma target 5.0
            #pragma multi_compile_instancing
            #pragma shader_feature_local _OVERLAY_MODE_MESH _OVERLAY_MODE_FULLSCREEN _OVERLAY_MODE_BILLBOARD_SPHERE
            #pragma instancing_options procedural:vertInstancingSetup

            #pragma vertex vertex_stage
            #pragma fragment fragment_stage
            
            #include "UnityCG.cginc"
            #include "UnityStandardParticleInstancing.cginc"
            #include "common.hlsl"

            struct VertexInput {
                float3 position_os : POSITION;
                float2 uv0 : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct FragmentInput {
                float4 position : SV_POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
                OverlayFragmentInputExtra overlay_extra;
            };
            struct FragmentOutput {
                half4 color : SV_Target;
                OverlayFragmentOutputExtra overlay_extra;
            };
            
            void vertex_stage (VertexInput input, uint vertex_id : SV_VertexID, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                setup_unity_birp_MatrixInvP();
                output.position = OverlayObjectToClipPos(input.position_os, input.uv0, vertex_id, output.overlay_extra);
            }

            void fragment_stage (FragmentInput input, out FragmentOutput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                setup_unity_birp_MatrixInvP();
                OverlayFragment(input.overlay_extra, output.overlay_extra);
                output.color = half4(0, 0, 0, 0);
            }
            ENDCG
        }
    }
}
