// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// Debug view for lighting metadata by generating 3D gizmos.
// Supported :
// - Dynamic lights : Directional, Point (pixel or vertex), Spotlight cone
// - Light probes : indirect diffuse on a sphere.
// - Reflection probes : solid boundary boxes, and dashed lines towards probe position. White/grey depending on blending. Texture value on a sphere.
// - Displays REDSIM VRC Light Volumes (V1.0) as dashed boxes. Dash count is the resolution.
// - LTCGI surfaces as colored quads.
// TODO REDSIM Light Volumes V2 when stabilised and released (point/spot lights, new metadata format)
// Not supported : Light Probe Proxy Volume ; this is a volume attached to a renderer that will provide the renderer with per-pixel light probes, so useless on avatars.

Shader "Lereldarion/Debug/Lighting" {
    Properties {
        _Distance_Limit ("Enable the debug view only if distance object-camera is lower than this threshold", Float) = 10
        _LightProbe_Radius ("Light Probe sphere radius", Float) = 0.05
        _ReflectionProbe_Radius ("Reflection Probe sphere radius", Float) = 0.2
    }
    SubShader {
        Tags {
            "Queue" = "Overlay"
            "RenderType" = "Overlay"
            "VRCFallback" = "Hidden"
            "PreviewType" = "Plane"
        }

        CGINCLUDE
        #pragma target 5.0

        #include "UnityCG.cginc"
        #include "UnityLightingCommon.cginc"
        #include "AutoLight.cginc"

        uniform float _VRChatMirrorMode;
        uniform float _Distance_Limit;
        uniform float _LightProbe_Radius;
        uniform float _ReflectionProbe_Radius;

        // Structs

        struct MeshData {
            UNITY_VERTEX_INPUT_INSTANCE_ID
        };
        
        struct LinePoint {
            float4 position : SV_POSITION;
            half3 color : LINE_COLOR;
            float dash : DASH;
            UNITY_VERTEX_OUTPUT_STEREO
        };

        struct Sphere {
            float4 position : SV_POSITION;
            float3 normal : NORMAL_WS;
            nointerpolation uint mode : MODE; // 0 = LightProbe, 1 = SpecCube0, 2 = SpecCube1
            UNITY_VERTEX_OUTPUT_STEREO
        };

        // Utils

        #if defined(USING_STEREO_MATRICES)
        static float3 centered_camera_ws = (unity_StereoWorldSpaceCameraPos[0] + unity_StereoWorldSpaceCameraPos[1]) / 2;
        #else
        static float3 centered_camera_ws = _WorldSpaceCameraPos.xyz;
        #endif

        float length_sq(float3 v) { return dot(v, v); }
        half length_sq(half3 v) { return dot(v, v); }

        float3x3 referential_from_z(float3 z) {
            z = normalize(z);
            float3 x = cross(z, float3(0, 1, 0)); // Cross with world up for the sides
            if(length_sq(x) == 0) {
                x = float3(1, 0, 0); // Fallback if aligned
            } else {
                x = normalize(x);
            }
            float3 y = cross(x, z);
            return float3x3(x, y, z);
        }

        // math adapted from https://gist.github.com/mairod/a75e7b44f68110e1576d77419d608786?permalink_comment_id=3180018#gistcomment-3180018
        // rotates in YIQ color space for efficiency
        half3 hue_shift_yiq(const half3 col, const half hueAngle) {
            const half3 k = 0.57735;
            const half sinAngle = sin(hueAngle);
            const half cosAngle = cos(hueAngle);
            return col * cosAngle + cross(k, col) * sinAngle + k * dot(k, col) * (1.0 - cosAngle);
        }

        float4x4 inverse(float4x4 mat) {
            // by lox9973
            float4x4 M=transpose(mat);
            float m01xy=M[0].x*M[1].y-M[0].y*M[1].x;
            float m01xz=M[0].x*M[1].z-M[0].z*M[1].x;
            float m01xw=M[0].x*M[1].w-M[0].w*M[1].x;
            float m01yz=M[0].y*M[1].z-M[0].z*M[1].y;
            float m01yw=M[0].y*M[1].w-M[0].w*M[1].y;
            float m01zw=M[0].z*M[1].w-M[0].w*M[1].z;
            float m23xy=M[2].x*M[3].y-M[2].y*M[3].x;
            float m23xz=M[2].x*M[3].z-M[2].z*M[3].x;
            float m23xw=M[2].x*M[3].w-M[2].w*M[3].x;
            float m23yz=M[2].y*M[3].z-M[2].z*M[3].y;
            float m23yw=M[2].y*M[3].w-M[2].w*M[3].y;
            float m23zw=M[2].z*M[3].w-M[2].w*M[3].z;
            float4 adjM0,adjM1,adjM2,adjM3;
            adjM0.x=+dot(M[1].yzw,float3(m23zw,-m23yw,m23yz));
            adjM0.y=-dot(M[0].yzw,float3(m23zw,-m23yw,m23yz));
            adjM0.z=+dot(M[3].yzw,float3(m01zw,-m01yw,m01yz));
            adjM0.w=-dot(M[2].yzw,float3(m01zw,-m01yw,m01yz));
            adjM1.x=-dot(M[1].xzw,float3(m23zw,-m23xw,m23xz));
            adjM1.y=+dot(M[0].xzw,float3(m23zw,-m23xw,m23xz));
            adjM1.z=-dot(M[3].xzw,float3(m01zw,-m01xw,m01xz));
            adjM1.w=+dot(M[2].xzw,float3(m01zw,-m01xw,m01xz));
            adjM2.x=+dot(M[1].xyw,float3(m23yw,-m23xw,m23xy));
            adjM2.y=-dot(M[0].xyw,float3(m23yw,-m23xw,m23xy));
            adjM2.z=+dot(M[3].xyw,float3(m01yw,-m01xw,m01xy));
            adjM2.w=-dot(M[2].xyw,float3(m01yw,-m01xw,m01xy));
            adjM3.x=-dot(M[1].xyz,float3(m23yz,-m23xz,m23xy));
            adjM3.y=+dot(M[0].xyz,float3(m23yz,-m23xz,m23xy));
            adjM3.z=-dot(M[3].xyz,float3(m01yz,-m01xz,m01xy));
            adjM3.w=+dot(M[2].xyz,float3(m01yz,-m01xz,m01xy));
            float invDet=rcp(dot(M[0].xyzw,float4(adjM0.x,adjM1.x,adjM2.x,adjM3.x)));
            return transpose(float4x4(adjM0*invDet,adjM1*invDet,adjM2*invDet,adjM3*invDet));
        }

        float3 arbitrary_anchor_point_ws() {
            // A point in front of the camera to attach items such as directional light vector, lightprobe values...
            float3 camera_fwd_ws = mul((float3x3) unity_MatrixInvV, float3(0, 0, -1));
            return centered_camera_ws + 1.5 * camera_fwd_ws;
        }

        bool within_distance_limit() {
            float3 object_ws = mul(unity_ObjectToWorld, float4(0, 0, 0, 1)).xyz;
            return length_sq(object_ws - centered_camera_ws) <= _Distance_Limit * _Distance_Limit;
        }

        // Drawing

        struct LineDrawer {
            LinePoint output;

            static LineDrawer init(half3 color) {
                LineDrawer drawer;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(drawer.output);
                drawer.output.color = color;
                drawer.output.dash = 0;
                return drawer;
            }

            void init_cs(inout LineStream<LinePoint> stream, float4 position_cs) {
                output.dash = 0;
                output.position = position_cs;
                stream.RestartStrip();
                stream.Append(output);
            }
            void init_ws(inout LineStream<LinePoint> stream, float3 position_ws) {
                init_cs(stream, UnityWorldToClipPos(position_ws));
            }

            void solid_cs(inout LineStream<LinePoint> stream, float4 position_cs) {
                output.position = position_cs;
                stream.Append(output);
            }
            void solid_ws(inout LineStream<LinePoint> stream, float3 position_ws) {
                solid_cs(stream, UnityWorldToClipPos(position_ws));
            }

            void dashed_cs(inout LineStream<LinePoint> stream, float4 position_cs, float dashes) {
                output.dash += dashes;
                output.position = position_cs;
                stream.Append(output);
            }
            void dashed_ws(inout LineStream<LinePoint> stream, float3 position_ws, float dashes) {
                dashed_cs(stream, UnityWorldToClipPos(position_ws), dashes);
            }
        };

        void directional_light(inout LineStream<LinePoint> s, half3 color, float3 light_forward_ws) {
            LineDrawer drawer = LineDrawer::init(color); // 10 calls
            float3x3 referential = referential_from_z(light_forward_ws);

            // Line to an arbitrary point in front of the camera (directional lights are infinite)
            float3 target = arbitrary_anchor_point_ws();
            float3 origin = target - 0.5 * referential[2];
            float4 origin_cs = UnityWorldToClipPos(origin);
            drawer.init_ws(s, target); drawer.solid_cs(s, origin_cs);

            // Tetrahedron on origin.
            float size = 0.05;
            float4 tetrahedron_a = UnityWorldToClipPos(origin + mul(size * float3(0, 1, -1.5), referential));
            float4 tetrahedron_b = UnityWorldToClipPos(origin + mul(size * float3(0.86, -0.5, -1.5), referential));
            float4 tetrahedron_c = UnityWorldToClipPos(origin + mul(size * float3(-0.86, -0.5, -1.5), referential));
            drawer.solid_cs(s, tetrahedron_a); drawer.solid_cs(s, tetrahedron_b);
            drawer.init_cs(s, origin_cs); drawer.solid_cs(s, tetrahedron_b); drawer.solid_cs(s, tetrahedron_c);
            drawer.init_cs(s, origin_cs); drawer.solid_cs(s, tetrahedron_c); drawer.solid_cs(s, tetrahedron_a);
        }

        void point_light(inout LineStream<LinePoint> s, half3 color, float3 pos) {
            LineDrawer drawer = LineDrawer::init(color); // 16 calls
            float3 ray_to_camera = centered_camera_ws - pos;
            float3x3 referential = referential_from_z(ray_to_camera);

            // 8 spokes
            float size = 0.05;
            [unroll]
            for(uint i = 0; i < 8; i += 1) {
                float3 spoke_ray = float3(0, 0, 0);
                sincos(i * UNITY_TWO_PI / 8, spoke_ray.x, spoke_ray.y);
                drawer.init_ws(s, pos + mul(size * spoke_ray, referential)); drawer.solid_ws(s, pos + mul(2 * size * spoke_ray, referential));
            }
        }

        void vertex_point_light(inout LineStream<LinePoint> s, half3 color, float3 pos) {
            LineDrawer drawer = LineDrawer::init(color); // 16 calls
            float3 ray_to_camera = centered_camera_ws - pos;
            float3x3 referential = referential_from_z(ray_to_camera);

            // 8 spokes with 2 upper ones forming a V for vertex
            float size = 0.05;
            [unroll]
            for(uint i = 0; i < 8; i += 1) {
                float3 spoke_ray = float3(0, 0, 0);
                sincos(i * UNITY_TWO_PI / 8, spoke_ray.x, spoke_ray.y);
                bool reach_center = i == 1 || i == 7;
                drawer.init_ws(s, pos + mul((reach_center ? 0 : size) * spoke_ray, referential)); drawer.solid_ws(s, pos + mul(2 * size * spoke_ray, referential));
            }
        }

        void spot_light(inout LineStream<LinePoint> s, half3 color, float3 pos, float4x4 world_to_light) {
            LineDrawer drawer = LineDrawer::init(color); // 20 vertexcount

            // W2L * (W.xyz, 1) = L.xyzw
            // From AutoLight.cginc : the light cone is defined by uv.xy=L.xy/L.w in [-0.5, 0.5] and L.z in [0, 1]
            // W2L.012 * W.xyz = (uv.xy * L.w, L.zw) - W2L.3 = L.w * (uv.xy, 0, 1) + (0, 0, L.z, 0) - W2L.3
            // [W2L.012|(-uv.xy, 0, -1)] * (W.xyz, L.w) = (0, 0, L.z, 0) - W2L.3
            const float4 w2l_3 = world_to_light._m03_m13_m23_m33;

            // L = (0, 0, 1)
            world_to_light._m03_m13_m23_m33 = float4(0, 0, 0, -1);
            float3 L001_ws = mul(inverse(world_to_light), float4(0, 0, 1, 0) - w2l_3).xyz;
            // +x : L = (0.5, 0, 1)
            world_to_light._m03_m13_m23_m33 = float4(0.5, 0, 0, -1);
            float3 Lx_ws = mul(inverse(world_to_light), float4(0, 0, 1, 0) - w2l_3).xyz - L001_ws;
            // +y : L = (0, 0.5, 1)
            world_to_light._m03_m13_m23_m33 = float4(0, 0.5, 0, -1);
            float3 Ly_ws = mul(inverse(world_to_light), float4(0, 0, 1, 0) - w2l_3).xyz - L001_ws;

            // Cone
            float4 cone_point_cs = UnityWorldToClipPos(pos);
            float4 cone_base_cs[8];
            [unroll]
            for(uint i = 0; i < 8; i += 1) {
                float2 scale;
                sincos(i * UNITY_TWO_PI / 8, scale.x, scale.y);
                cone_base_cs[i] = UnityWorldToClipPos(L001_ws + scale.x * Lx_ws + scale.y * Ly_ws);
            }
            for(i = 0; i < 8; i += 2) {
                drawer.init_cs(s, cone_base_cs[i + 1]); drawer.solid_cs(s, cone_base_cs[i]);
                drawer.solid_cs(s, cone_point_cs); drawer.solid_cs(s, cone_base_cs[i + 1]);
                drawer.solid_cs(s, cone_base_cs[(i + 2) % 8]);
            }
        }

        void reflexion_probe(inout LineStream<LinePoint> s, half3 color, float3 position_ws, float3 bbox_min, float3 bbox_max) {
            LineDrawer drawer = LineDrawer::init(color); // 21 calls
            // With no defined reflection boxes, unity will use the skybox with infinite bounding boxes.
            bool is_skybox = any(!isfinite(bbox_min));
            if(is_skybox) {
                bbox_min = position_ws - _Distance_Limit;
                bbox_max = position_ws + _Distance_Limit;
            }
            // Precompute center and 8 corners (x = +1, y = +2, z = +4)
            float4 center = UnityWorldToClipPos(position_ws);
            float dashes[8];
            float4 corners[8];
            for(uint i = 0; i < 8; i += 1) {
                float3 corner = i & uint3(1, 2, 4) ? bbox_max : bbox_min;
                dashes[i] = 10 * round(distance(corner, position_ws));
                corners[i] = UnityWorldToClipPos(corner);
            }
            // Drawing
            if(!is_skybox) {
                // Draw cube and lines from corner to center. In one big line.
                drawer.init_cs(s, center);
                drawer.dashed_cs(s, corners[0], dashes[0]); drawer.solid_cs(s, corners[1]); drawer.solid_cs(s, corners[5]); drawer.solid_cs(s, corners[4]); drawer.dashed_cs(s, center, dashes[4]);
                drawer.dashed_cs(s, corners[1], dashes[1]); drawer.solid_cs(s, corners[3]); drawer.solid_cs(s, corners[7]); drawer.solid_cs(s, corners[5]); drawer.dashed_cs(s, center, dashes[5]);
                drawer.dashed_cs(s, corners[3], dashes[3]); drawer.solid_cs(s, corners[2]); drawer.solid_cs(s, corners[6]); drawer.solid_cs(s, corners[7]); drawer.dashed_cs(s, center, dashes[7]);
                drawer.dashed_cs(s, corners[2], dashes[2]); drawer.solid_cs(s, corners[0]); drawer.solid_cs(s, corners[4]); drawer.solid_cs(s, corners[6]); drawer.dashed_cs(s, center, dashes[6]);
            } else {
                // Just draw corner-center lines
                drawer.init_cs(s, corners[0]); drawer.dashed_cs(s, corners[7], dashes[0] + dashes[7]);
                drawer.init_cs(s, corners[1]); drawer.dashed_cs(s, corners[6], dashes[1] + dashes[6]);
                drawer.init_cs(s, corners[2]); drawer.dashed_cs(s, corners[5], dashes[2] + dashes[5]);
                drawer.init_cs(s, corners[3]); drawer.dashed_cs(s, corners[4], dashes[3] + dashes[4]);
            }
        }

        // VRC Light Volumes https://github.com/REDSIM/VRCLightVolumes/

        uniform float _UdonLightVolumeCount; // All volumes count in scene
        uniform Texture3D _UdonLightVolume; // Main 3D Texture atlas
        uniform float4x4 _UdonLightVolumeInvWorldMatrix[32]; // World to Local (-0.5, 0.5) UVW Matrix
        uniform float3 _UdonLightVolumeUvw[192]; // AABB Bounds of islands on the 3D Texture atlas

        void draw_vrc_light_volume(inout LineStream<LinePoint> s, uint volume_id) {
            if(!((float) volume_id < min(_UdonLightVolumeCount, 32))) { return ; }

            // Select a color for the volume ; hue shift gradient from red.
            half3 color = hue_shift_yiq(half3(1, 0, 0), volume_id / _UdonLightVolumeCount * UNITY_TWO_PI);
            LineDrawer drawer = LineDrawer::init(color); // 20 calls

            // Project bounds to CS
            float4x4 volume_to_world = inverse(_UdonLightVolumeInvWorldMatrix[volume_id]);
            float4x4 volume_to_cs = mul(UNITY_MATRIX_VP, volume_to_world);
            float4 corners[8];
            for(uint i = 0; i < 8; i += 1) {
                float3 corner = i & uint3(1, 2, 4) ? -0.5 : 0.5;
                corners[i] = mul(volume_to_cs, float4(corner, 1));
            }

            // Texel resolution as dash count
            float3 atlas_texels;
            _UdonLightVolume.GetDimensions(atlas_texels.x, atlas_texels.y, atlas_texels.z);
            uint uvwID = volume_id * 6;
            float3 uvw_resolution = _UdonLightVolumeUvw[uvwID + 1].xyz - _UdonLightVolumeUvw[uvwID].xyz;
            float3 texels = atlas_texels * uvw_resolution;

            // Draw edges with texel count
            drawer.init_cs(s, corners[0]); drawer.dashed_cs(s, corners[1], texels.x); drawer.dashed_cs(s, corners[5], texels.z); drawer.dashed_cs(s, corners[4], texels.x);
            drawer.init_cs(s, corners[1]); drawer.dashed_cs(s, corners[3], texels.y); drawer.dashed_cs(s, corners[7], texels.z); drawer.dashed_cs(s, corners[5], texels.y);
            drawer.init_cs(s, corners[3]); drawer.dashed_cs(s, corners[2], texels.x); drawer.dashed_cs(s, corners[6], texels.z); drawer.dashed_cs(s, corners[7], texels.x);
            drawer.init_cs(s, corners[2]); drawer.dashed_cs(s, corners[0], texels.y); drawer.dashed_cs(s, corners[4], texels.z); drawer.dashed_cs(s, corners[6], texels.y);
        }

        // LTCGI https://github.com/PiMaker/ltcgi/

        uniform uint _Udon_LTCGI_ScreenCount; // Up to 16
        uniform bool _Udon_LTCGI_Mask[16];
        uniform float4 _Udon_LTCGI_Vertices_0[16];
        uniform float4 _Udon_LTCGI_Vertices_1[16];
        uniform float4 _Udon_LTCGI_Vertices_2[16];
        uniform float4 _Udon_LTCGI_Vertices_3[16];

        void draw_ltcgi_surface(inout LineStream<LinePoint> s, uint screen_id) {
            if(!(screen_id < min(_Udon_LTCGI_ScreenCount, 16) && !_Udon_LTCGI_Mask[screen_id])) { return; }

            // Select a color for the volume ; hue shift gradient from red.
            half3 color = hue_shift_yiq(half3(1, 0, 0), (screen_id * UNITY_TWO_PI) / _Udon_LTCGI_ScreenCount);
            LineDrawer drawer = LineDrawer::init(color); // 7 calls

            float3 v0 = _Udon_LTCGI_Vertices_0[screen_id].xyz;
            float3 v1 = _Udon_LTCGI_Vertices_1[screen_id].xyz;
            float3 v2 = _Udon_LTCGI_Vertices_2[screen_id].xyz;
            float3 v3 = _Udon_LTCGI_Vertices_3[screen_id].xyz;

            // Alternate way to get positions. Both works.
            // Rotating screens on Pi's map are stuck not rotating in both modes...
            // uniform Texture2D<float4> _Udon_LTCGI_static_uniforms;
            // float3 v0 = _Udon_LTCGI_static_uniforms[uint2(0, screen_id)].xyz;
            // float3 v1 = _Udon_LTCGI_static_uniforms[uint2(1, screen_id)].xyz;
            // float3 v2 = _Udon_LTCGI_static_uniforms[uint2(2, screen_id)].xyz;
            // float3 v3 = _Udon_LTCGI_static_uniforms[uint2(3, screen_id)].xyz;

            float3 center = (v0 + v1 + v2 + v3) / 4;
            float3 normal = normalize(cross(v1 - v0, v2 - v0)) * -1; // -1 = cross is defined for vector from geom to screen

            drawer.init_ws(s, v0); drawer.solid_ws(s, v1); drawer.solid_ws(s, v3); drawer.solid_ws(s, v2); drawer.solid_ws(s, v0);
            drawer.init_ws(s, center); drawer.solid_ws(s, center + normal * length(v0 - center)); // Normal vector
        }

        // Stages for line gizmos

        void vertex_stage (MeshData input, out MeshData output) {
            output = input;
        }

        half4 fragment_lines_visible (LinePoint line_point) : SV_Target {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(line_point);

            // Dashing : line and voids of length 1 along dash_counter
            float dash_01 = frac(line_point.dash * 0.5);
            if (dash_01 > 0.5 && dash_01 < 0.99) { discard; }

            return half4(line_point.color, 1);
        }
        half4 fragment_lines_occluded (LinePoint line_point) : SV_Target {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(line_point);

            // Dashing : line and voids of length 1 along dash_counter
            float dash_01 = frac(line_point.dash * 0.5);
            if (dash_01 > 0.5 && dash_01 < 0.99) { discard; }

            return half4(line_point.color, 0.1);
        }

        [instance(32)]
        [maxvertexcount(41)]
        void geometry_lines_base(point MeshData input[1], uint primitive_id : SV_PrimitiveID, uint instance : SV_GSInstanceID, inout LineStream<LinePoint> stream) {
            UNITY_SETUP_INSTANCE_ID(input[0]);

            if (!(_VRChatMirrorMode == 0 && primitive_id == 0 && within_distance_limit())) { return; }

            // First try drawing high count items.
            draw_vrc_light_volume(stream, instance); // 32 Volumes, 1 per instance. 20 vertexcount
            draw_ltcgi_surface(stream, instance); // 16 Screens, 1 per instance for first 16. 5 vertexcount

            // Spread other workload on threads, starting at the end in case volumes are not using all instances.
            [forcecase]
            switch (instance) {
                case 31: {
                    // Main directional light, which may be disabled
                    float3 light_forward_ws = -_WorldSpaceLightPos0.xyz;
                    if(length_sq(light_forward_ws) > 0) {
                        directional_light(stream, _LightColor0.rgb, light_forward_ws);
                    }
                    break;
                }
                case 30: case 29: {
                    // Reflexion probes : 21 vertexcount
                    bool is_primary = instance == 30;
                    #if defined(UNITY_SPECCUBE_BLENDING) || defined(FORCE_BOX_PROJECTION)
                    float blend_factor = is_primary ? unity_SpecCube0_BoxMin.w : 1 - unity_SpecCube0_BoxMin.w;
                    #else
                    float blend_factor = is_primary ? 1 : 0;
                    #endif
                    float3 position = is_primary ? unity_SpecCube0_ProbePosition.xyz : unity_SpecCube1_ProbePosition.xyz;
                    float3 bbox_min = is_primary ? unity_SpecCube0_BoxMin.xyz : unity_SpecCube1_BoxMin.xyz;
                    float3 bbox_max = is_primary ? unity_SpecCube0_BoxMax.xyz : unity_SpecCube1_BoxMax.xyz;
                    bool ignore_secondary = !is_primary && unity_SpecCube0_BoxMin.w >= 0.99999; // Ignore secondary if not used according to blend factor
                    if (blend_factor > 0.00001) {
                        reflexion_probe(stream, blend_factor * half3(1, 1, 1), position, bbox_min, bbox_max);
                    }
                    break;
                }
                case 28: case 27: case 26: case 25: {
                    // 4 Vertex point lights : 16 vertexcount
                    uint vertex_light_id = 28 - instance;
                    half3 color = unity_LightColor[vertex_light_id].rgb;
                    if(length_sq(color) > 0) {
                        float3 position = float3(unity_4LightPosX0[vertex_light_id], unity_4LightPosY0[vertex_light_id], unity_4LightPosZ0[vertex_light_id]);
                        vertex_point_light(stream, color, position);
                    }
                    break;
                }
                default: break;
            }
        }

        [maxvertexcount(20)]
        void geometry_lines_add(point MeshData input[1], uint primitive_id : SV_PrimitiveID, inout LineStream<LinePoint> stream) {
            UNITY_SETUP_INSTANCE_ID(input[0]);

            if (!(_VRChatMirrorMode == 0 && primitive_id == 0 && within_distance_limit())) { return; }

            #if defined(POINT) || defined(POINT_COOKIE)
            point_light(stream, _LightColor0.rgb, _WorldSpaceLightPos0.xyz);
            #elif defined(DIRECTIONAL) || defined(DIRECTIONAL_COOKIE)
            directional_light(stream, _LightColor0.rgb, -_WorldSpaceLightPos0.xyz);
            #elif defined(SPOT)
            spot_light(stream, _LightColor0.rgb, _WorldSpaceLightPos0.xyz, unity_WorldToLight);
            #endif
        }

        // Sphere functions

        static const uint sphere_subdivision = 4;

        void draw_sphere_patch(inout TriangleStream<Sphere> stream, uint mode, uint id, float3 center, float radius) {
            // 6 faces, 4 strips of 4 quads for a cube-topology sphere.
            uint face_id = id / sphere_subdivision; // [0, 6[
            uint main_axis_id = face_id >= 3 ? face_id - 3 : face_id; // 0,1,2
            float strip_id = id - sphere_subdivision * face_id; // [0, subdiv[
            // Select axis for the face construction. normal is always active, and uv go from [-1, 1]
            float3 normal = main_axis_id == uint3(0, 1, 2) ? (face_id >= 3 ? -1 : 1) : 0; // 6 orientations
            float3 axis_u = main_axis_id == uint3(2, 0, 1) ? 1 : 0; // Any other axis
            float3 axis_v = cross(normal, axis_u); // Last vector, use cross for consistent triangle winding
            // Strip along v
            float3 a_u_normal_components = normal + lerp(-axis_u, axis_u, strip_id / float(sphere_subdivision));
            float3 b_u_normal_components = normal + lerp(-axis_u, axis_u, (strip_id + 1) / float(sphere_subdivision));
            Sphere sphere;
            UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(sphere);
            sphere.mode = mode;
            stream.RestartStrip();
            for(uint i = 0; i <= sphere_subdivision; i += 1) {
                float3 v_component = lerp(-axis_v, axis_v, i / float(sphere_subdivision));
                sphere.normal = normalize(a_u_normal_components + v_component) * radius;
                sphere.position = UnityWorldToClipPos(center + sphere.normal);
                stream.Append(sphere);
                sphere.normal = normalize(b_u_normal_components + v_component) * radius;
                sphere.position = UnityWorldToClipPos(center + sphere.normal);
                stream.Append(sphere);
            }
        }

        half4 fragment_sphere (Sphere sphere) : SV_Target {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(sphere);
            float3 normal = normalize(sphere.normal);
            switch(sphere.mode) {
                case 0: return half4(ShadeSH9(float4(normal, 1)), 1);
                case 1: return half4(DecodeHDR(UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, normal, 0), unity_SpecCube0_HDR), 1);
                case 2: return half4(DecodeHDR(UNITY_SAMPLE_TEXCUBE_SAMPLER_LOD(unity_SpecCube1, unity_SpecCube0, normal, 0), unity_SpecCube1_HDR), 1);
                default: return half4(0, 0, 0, 1);
            }
        }

        [instance(6 * sphere_subdivision)]
        [maxvertexcount((sphere_subdivision + 1) * 2 * 3)]
        void geometry_spheres(point MeshData input[1], uint primitive_id : SV_PrimitiveID, uint instance : SV_GSInstanceID, inout TriangleStream<Sphere> stream) {
            UNITY_SETUP_INSTANCE_ID(input[0]);

            if (!(_VRChatMirrorMode == 0 && primitive_id == 0 && within_distance_limit())) { return; }

            draw_sphere_patch(stream, 0, instance, arbitrary_anchor_point_ws(), _LightProbe_Radius); // Light Probe
            draw_sphere_patch(stream, 1, instance, unity_SpecCube0_ProbePosition, _ReflectionProbe_Radius);
            #if defined(UNITY_SPECCUBE_BLENDING) || defined(FORCE_BOX_PROJECTION)
            if (unity_SpecCube0_BoxMin.w < 0.99999) {
                draw_sphere_patch(stream, 2, instance, unity_SpecCube1_ProbePosition, _ReflectionProbe_Radius);
            }
            #endif
        }

        ENDCG

        Pass {
            Name "Base Lighting Visible Lines"
            Tags { "LightMode" = "ForwardBase" }

            ZTest LEqual

            CGPROGRAM
            #pragma vertex vertex_stage
            #pragma geometry geometry_lines_base
            #pragma fragment fragment_lines_visible
            #pragma multi_compile_instancing
            ENDCG
        }
        Pass {
            Name "Additive Lighting Visible Lines"
            Tags { "LightMode" = "ForwardAdd" }

            ZTest LEqual

            CGPROGRAM
            #pragma vertex vertex_stage
            #pragma geometry geometry_lines_add
            #pragma fragment fragment_lines_visible
            #pragma multi_compile_instancing
            #pragma multi_compile_fwdadd
            ENDCG
        }
        Pass {
            Name "Base Lighting Spheres"
            Tags { "LightMode" = "ForwardBase" }

            ZTest LEqual

            CGPROGRAM
            #pragma vertex vertex_stage
            #pragma geometry geometry_spheres
            #pragma fragment fragment_sphere
            #pragma multi_compile_instancing
            ENDCG
        }
        Pass {
            Name "Base Lighting Occluded Lines"
            Tags { "LightMode" = "ForwardBase" }

            ZTest Greater
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off

            CGPROGRAM
            #pragma vertex vertex_stage
            #pragma geometry geometry_lines_base
            #pragma fragment fragment_lines_occluded
            #pragma multi_compile_instancing
            ENDCG
        }
        Pass {
            Name "Additive Lighting Occluded Lines"
            Tags { "LightMode" = "ForwardAdd" }

            ZTest Greater
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off

            CGPROGRAM
            #pragma vertex vertex_stage
            #pragma geometry geometry_lines_add
            #pragma fragment fragment_lines_occluded
            #pragma multi_compile_instancing
            #pragma multi_compile_fwdadd
            ENDCG
        }
    }
}
