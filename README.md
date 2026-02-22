# Lereldarion Unity Shaders
Collection of useful Unity Shaders, made for VRChat.
They are not dependent on its SDK but some features rely on [VRChat shader globals](https://creators.vrchat.com/worlds/udon/vrc-graphics/vrchat-shader-globals/).
Unless specified, they have only be tested in a PC VR VRChat context : DX11, unity built-in render pipeline.

Install :
- using VCC at https://lereldarion.github.io/vpm-listing/.
- or from Unity Packages from [releases](https://github.com/lereldarion/unity-shaders/releases).

[Github repository](https://github.com/lereldarion/unity-shaders/) in case you found this documentation from an installed package.

## Overlays
These overlay shaders are useful on avatars for world analysis (wireframe, normals, grid, HUD), or adjusting lighting (GammaAdjust).

All overlays have multiple modes, selected by `_Overlay_Mode` property :
- `Mesh` : applied on the mesh
- `Fullscreen` : applied as a screenspace shader (fullscreen).
- `Billboard Sphere` : emulate a sphere mesh, with fullscreen when camera is inside.
  Support mesh **must** be a flat surface with uniform normal. The sphere is defined by `UV0` : center at `(0.5, 0.5)`, radius `0.5`. Example of valid mesh : unity quad.

If you use a fullscreen mode, it is good practice to only animate the mode to fullscreen **locally** (`IsLocal` VRChat parameter) to avoid annoying others.
Fullscreen mode uses [VRChat shader globals](https://creators.vrchat.com/worlds/udon/vrc-graphics/vrchat-shader-globals/) to ignore mirror (always) and secondary cameras (toggle).
Note that this fullscreen effect only works if the mesh renderer is not culled by Unity, so ensure that the mesh bounds are in view of the player camera.

Since v1.4.0 the fullscreen mode uses the vertex stage only (no geometry pass).
This will fail depending on the mesh vertex ordering ; symptoms are a missing triangle on the screen.
To fix it change the `vertex order` slider until it displays correctly.

The `Demo` scene demonstrate their use on a dummy scene.

### Gamma Adjust
Adjust the gamma of the scene image behind it, like a post process effect.
Uses a `GrabPass`.

This is very useful to simulate low-light vision in dark spaces (without a dynamic light !), or decrease luminosity in worlds that are too bright.

![](.github/overlay_gamma_adjust.jpg)

### Depth Scene Reconstruction : Grid / Normals / Wireframe
These 3 overlays reconstruct the position of the scene pixels, and use it for various display effects :
- `Grid` overlays a 1m (dependent on Zoom) world space grid on the scene objects, along X/Y/Z planes
- `Normals` shows the world space normals of the actual triangle geometry of the scene (without the normal maps !), color coded
- `Wireframe` displays scene triangles with white edges on dark

All of these overlays use the `_CameraDepthTexture`, a unity feature to access the depth of scene pixels.
This texture is only available if required by the render pipeline, usually if a realtime dynamic light with shadows is present on the scene.
For worlds without dynamic light and shadows, you can force `_CameraDepthTexture` by adding a light on your avatar at the cost of avatar Rank : https://github.com/netri/Neitri-Unity-Shaders?tab=readme-ov-file#types.
I personnaly choose to not put a light on my avatars for performance reasons (they are really bad !), and accept that the `_CameraDepthTexture` is not always available.

These shaders are inspired by similar ones from https://github.com/netri/Neitri-Unity-Shaders, but they have been improved to be more numerically precise and do not use any `GrabPass`.

#### Grid
![](.github/overlay_grid.jpg)

#### Normals
![](.github/overlay_normals.jpg)

#### Wireframe
![](.github/overlay_wireframe.jpg)

### HUD
This shader shows various world positionning data in a fighter jet like HUD with emission :
- world position X/Y/Z in meters
- world compass : azimuth and elevation
- a range in meters to the object pointed by the central crosshair, if the `_CameraDepthTexture` is available (see note for depth).
- camera far plane distance
- local fps

The HUD displays like a skybox that is anchored to the object on which the shader is applied ; like a [reflector sight](https://en.wikipedia.org/wiki/Reflector_sight).
The central crosshair will we aligned with the surface normal vector.
This works well on a simple quad.
For more complex geometries, you must ensure that all triangles have the same normal vectors ; this can be done with custom normals in blender.

This shader uses a small internal texture for the font (MSDF strategy).

![](.github/overlay_hud.jpg)

## Debug tools
Other debug / introspection tools which are not overlays

### Lighting
Apply to any mesh (only 1 vertex required), and it will display lighting configuration elements that touch the mesh renderer as gizmos in the world :
- Unity realtime lights : pixel directional / point / spot lights, vertex point lights, colored by the light color.
- Unity reflection probes : displays the bounding boxes and their center, with a sphere to show the raw cubemap. Color of the lines shows the blend factor when 2 probes are active.
- Unity light probes : sphere with lighting values applied to it. Positions are not available in the shader so they cannot be displayed.
- [LTCGI](https://github.com/PiMaker/ltcgi) v1 : emitting surfaces as rectangle edges + diagonals. A normal line at center. Solid lines. Color is item index.
- [Light Volumes](https://github.com/REDSIM/VRCLightVolumes) v2 :
    - volume bounding boxes, dashed lines (dash size is texture resolution). Color hue represents indexes.
    - lights with simpler dashed patterns compared to unity equivalents. Color is the light color. Dash size is 1m and line lengths indicate culling distance. Display light cookies on spheres if used.

![](.github/debug_lighting.png)

### TBN
Applied to mesh, this shaders displays the tangent space at each vertex as small 3D lines.

![](.github/debug_tbn.png)

## Hidden
Material that does not render anything.

This is useful to fill material slots that cannot be disabled by animators on avatars (skinned mesh slots).

This shader comes with a material configured to use the [Material.SetShaderPassEnabled](https://docs.unity3d.com/ScriptReference/Material.SetShaderPassEnabled.html) trick by d4rkpl4y3r (https://github.com/d4rkc0d3r/UnityAndVRChatQuirks?tab=readme-ov-file#skipping-draw-call-for-material-slot).
If this is used properly, the material does not generate any draw call.
As a fallback, this sets the clip space positions to `NaN` to discard all geometry.