// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// An overlay which amplifies lighting by using a gamma curve with fractional exponent.
// Added Fullscreen "screenspace" mode, billboard sphere mode, radial dissolve effect

Shader "Lereldarion/Overlay/GammaAdjust" {
    Properties {
        [Header(Gamma)]
        _Gamma_Adjust_Value("Gamma Adjust Value", Range(-5, 5)) = 0
        [ToggleUI] _Transmit_Emission("Keep pixel values above 1 (emisison / bloom)", Float) = 1

        [Header(Overlay)]
        [KeywordEnum(Mesh, Fullscreen, Billboard Sphere)] _Overlay_Mode("Overlay mode", Float) = 0
        [IntRange] _Overlay_Fullscreen_Vertex_Order("Fullscreen vertex order (mesh dependent)", Range(0, 2)) = 0
        [ToggleUI] _Overlay_Fullscreen_Only_Main_Camera("Fullscreen mode restricted to main camera", Float) = 1
        [Enum(Surface Only, 0, Filled, 1)] _Overlay_Sphere_Filled("Sphere type", Float) = 1
        [Toggle(_OVERLAY_RADIAL_DISSOLVE_ENABLED)] _Overlay_Radial_Dissolve("Enable radial dissolve effect", Float) = 0
        _Overlay_Radial_Dissolve_Noise_Texture("Noise texture for dissolve", 2D) = "" {}
        _Overlay_Radial_Dissolve_Bounds("Radial dissolve bounds (radius start, end)", Vector) = (0.8, 1, 0, 0)
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
        
        Cull Off
        ZWrite On
        ZTest Less

        GrabPass { "_GammaAdjustGrabTexture" }

        Pass {
            CGPROGRAM
            #pragma target 5.0
            #pragma multi_compile_instancing
            #pragma shader_feature_local _OVERLAY_MODE_MESH _OVERLAY_MODE_FULLSCREEN _OVERLAY_MODE_BILLBOARD_SPHERE
            #pragma shader_feature_local __ _OVERLAY_RADIAL_DISSOLVE_ENABLED
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
                sample float4 position : SV_POSITION; // Explicit interpolation modifier required here
                float4 grab_screen_pos : GRAB_SCREEN_POS;
                nointerpolation float gamma : GAMMA;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
                OverlayFragmentInputExtra overlay_extra;
            };
            struct FragmentOutput {
                half4 color : SV_Target;
                OverlayFragmentOutputExtra overlay_extra;
            };
            
            uniform float _Gamma_Adjust_Value;
            uniform float _Transmit_Emission;
            
            UNITY_DECLARE_TEX2D(_GammaAdjustGrabTexture);

            void vertex_stage (VertexInput input, uint vertex_id : SV_VertexID, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                setup_unity_birp_MatrixInvP();

                output.position = OverlayObjectToClipPos(input.position_os, input.uv0, vertex_id, output.overlay_extra);
                output.grab_screen_pos = ComputeGrabScreenPos(output.position);
                output.gamma = exp(_Gamma_Adjust_Value); // exp(3 * (0.3 - _Gamma_Adjust_Value));
            }
            
            void fragment_stage (FragmentInput input, out FragmentOutput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                setup_unity_birp_MatrixInvP();

                OverlayFragment(input.overlay_extra, output.overlay_extra);

                const half4 scene_color = UNITY_SAMPLE_TEX2D_LOD(_GammaAdjustGrabTexture, input.grab_screen_pos.xy / input.grab_screen_pos.w, 0); // No mipmap as we take matching pixels
                const half3 clamped_color = saturate(scene_color.rgb); // Avoid screen explosion at positive gamma + emission (>1) + bloom.
                const half3 emission = _Transmit_Emission ? scene_color - clamped_color : 0;  
                output.color = half4(pow(clamped_color, input.gamma) + emission, 1);
            }

            ENDCG
        }
    }
}
