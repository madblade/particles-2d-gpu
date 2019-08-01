function createShader(gl, type, source) {
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
}

function createProgram(gl, vertexSource, fragmentSource) {
    var program = gl.createProgram();
    var vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
    var fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program));
    }
    var wrapper = {program: program};
    var numAttributes = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
    for (var i = 0; i < numAttributes; i++) {
        var attribute = gl.getActiveAttrib(program, i);
        wrapper[attribute.name] = gl.getAttribLocation(program, attribute.name);
    }
    var numUniforms = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (var i$1 = 0; i$1 < numUniforms; i$1++) {
        var uniform = gl.getActiveUniform(program, i$1);
        wrapper[uniform.name] = gl.getUniformLocation(program, uniform.name);
    }

    return wrapper;
}

function createTexture(gl, filter, data, width, height) {
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    if (data instanceof Uint8Array) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    } else {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, data);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
}

function bindTexture(gl, texture, unit) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
}

function createBuffer(gl, data) {
    var buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return buffer;
}

function bindAttribute(gl, buffer, attribute, numComponents) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(attribute);
    gl.vertexAttribPointer(attribute, numComponents, gl.FLOAT, false, 0, 0);
}

function bindFramebuffer(gl, framebuffer, texture) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    if (texture) {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    }
}

var drawVert = "precision mediump float;\n\n" +
    "attribute float a_index;\n\n" +
    "uniform sampler2D u_particles;\n" +
    "uniform float u_particles_res;\n\n" +
    "varying vec2 v_particle_pos;\n\n" +
    "void main() {\n" +
    "    vec4 color = texture2D(u_particles, vec2(\n" +
    "        fract(a_index / u_particles_res),\n" +
    "        floor(a_index / u_particles_res) / u_particles_res));\n\n" +
    "    // decode current particle position from the pixel's RGBA value\n" +
    "    v_particle_pos = vec2(\n" +
    "        color.r / 255.0 + color.b,\n" +
    "        color.g / 255.0 + color.a);\n\n" +
    "    gl_PointSize = 1.5;\n" +
    "    gl_Position = vec4(2.0 * v_particle_pos.x - 1.0, 1.0 - 2.0 * v_particle_pos.y, 0, 1);\n" +
    "}\n";

var drawFrag = "precision mediump float;\n\n" +
    "uniform mat4 luc;\n" +
    "uniform vec4 lucQ;\n" +
    "uniform vec4 lucL;\n" +
    "uniform sampler2D u_wind;\n" +
    "uniform vec2 u_wind_res;\n" +
    "uniform vec2 u_wind_min;\n" +
    "uniform vec2 u_wind_max;\n" +
    "uniform sampler2D u_color_ramp;\n\n" +
    "varying vec2 v_particle_pos;\n\n" +
    "void main() {\n" +
    "    float facX = u_wind_res.x > u_wind_res.y ? u_wind_res.x / u_wind_res.y : 1.0;\n" +
    "    float facY = u_wind_res.x > u_wind_res.y ? 1.0 : u_wind_res.y / u_wind_res.x;\n" +
    "    \n" +
    "    // compute for 8 tracers\n" +
    "    vec2 vel = vec2(0.0, 0.0);\n" +
    "    float gamma = 0.01;\n" +
    "    float epsilon = 2.0;\n" +
    "    for (int i = 0; i < 4; ++i) {\n" +
    "        float w1 = lucQ[i];\n" +
    "        float delta0 = facX * (luc[i][0] - v_particle_pos[0]);\n" +
    "        float delta1 = facY * (luc[i][1] - v_particle_pos[1]);\n" +
    "        float d2 = delta0 * delta0 + delta1 * delta1;\n" +
    "        float extinction = exp(-d2 / gamma);\n" +
    "        vel[0] += extinction * delta1 * epsilon * w1;\n" +
    "        vel[1] += extinction * (delta0 * epsilon * w1);\n" +
    "\n" +
    "        w1 = lucL[i];\n" +
    "        delta0 = facX * (luc[i][2] - v_particle_pos[0]);\n" +
    "        delta1 = facY * (luc[i][3] - v_particle_pos[1]);\n" +
    "        d2 = delta0 * delta0 + delta1 * delta1;\n" +
    "        extinction = exp(-d2 / gamma);\n" +
    "        vel[0] += extinction * delta1 * epsilon * w1;\n" +
    "        vel[1] += extinction * (delta0 * epsilon * w1);\n" +
    "    \n" +
    "    }\n" +
    "    vel[0] += 0.05;\n" +
    "    vec2 velocity = mix(u_wind_min, u_wind_max, vel);\n" +
    "    float speed_t = 8.0 * length(velocity) / length(u_wind_max);\n\n" +
    "    // color ramp is encoded in a 16x16 texture\n" +
    "    vec2 ramp_pos = vec2(\n" +
    "        fract(16.0 * speed_t),\n" +
    "        floor(16.0 * speed_t) / 16.0);\n\n" +
    "    vec4 outputColor = texture2D(u_color_ramp, ramp_pos);\n" +
    "    gl_FragColor = outputColor;\n" +
    "}\n";

var quadVert = "precision mediump float;\n\n" +
    "attribute vec2 a_pos;\n\n" +
    "varying vec2 v_tex_pos;\n\n" +
    "void main() {\n" +
    "    v_tex_pos = a_pos;\n" +
    "    gl_Position = vec4(1.0 - 2.0 * a_pos, 0, 1);\n" +
    "}\n";

var screenFrag = "precision mediump float;\n\n" +
    "uniform sampler2D u_screen;\n" +
    "uniform float u_opacity;\n\n" +
    "varying vec2 v_tex_pos;\n\n" +
    "void main() {\n" +
    "    vec4 color = texture2D(u_screen, 1.0 - v_tex_pos);\n" +
    "    // a hack to guarantee opacity fade out even with a value close to 1.0\n" +
    "    gl_FragColor = vec4(floor(255.0 * color * u_opacity) / 255.0);\n" +
    "}\n";

var updateFrag = "precision highp float;\n\n" +
    "uniform sampler2D u_particles;\n" +
    "uniform sampler2D u_wind;\n" +
    "uniform vec4 lucQ;\n" +
    "uniform vec4 lucL;\n" +
    "uniform mat4 luc;\n" +
    "uniform vec2 u_wind_res;\n" +
    "uniform vec2 u_wind_min;\n" +
    "uniform vec2 u_wind_max;\n" +
    "uniform float u_rand_seed;\n" +
    "uniform float u_speed_factor;\n" +
    "uniform float u_drop_rate;\n" +
    "uniform float u_drop_rate_bump;\n\n" +
    "varying vec2 v_tex_pos;\n\n" +
    "// pseudo-random generator\n" +
    "const vec3 rand_constants = vec3(12.9898, 78.233, 4375.85453);\n" +
    "float rand(const vec2 co) {\n" +
    "    float t = dot(rand_constants.xy, co);\n" +
    "    return fract(sin(t) * (rand_constants.z + t));\n" +
    "}\n" +
    "\n" +
    "// wind speed lookup; use manual bilinear filtering based on 4 adjacent pixels for smooth interpolation\n" +
    "vec2 lookup_wind(const vec2 uv) {\n" +
    "    // return texture2D(u_wind, uv).rg; // lower-res hardware filtering\n" +
    "    vec2 px = 1.0 / u_wind_res;\n" +
    "    vec2 vc = (floor(uv * u_wind_res)) * px;\n" +
    // "    vec2 f = fract(uv * u_wind_res);\n" +
    // "    vec2 tl = texture2D(u_wind, vc).rg;\n" +
    // "    vec2 tr = texture2D(u_wind, vc + vec2(px.x, 0)).rg;\n" +
    // "    vec2 bl = texture2D(u_wind, vc + vec2(0, px.y)).rg;\n" +
    // "    vec2 br = texture2D(u_wind, vc + px).rg;\n" +
    // "    return mix(mix(tl, tr, f.x), mix(bl, br, f.x), f.y);\n" +
    "    float facX = u_wind_res.x > u_wind_res.y ? u_wind_res.x / u_wind_res.y : 1.0;\n" +
    "    float facY = u_wind_res.x > u_wind_res.y ? 1.0 : u_wind_res.y / u_wind_res.x;\n" +
    "    vec2 n = normalize(u_wind_res);\n" +
    "    float gamma = 0.02 * max(n.x, n.y);\n" +
    "    float epsilon = -2.0;\n" +
    "    vec2 vel = vec2(0.0, 0.0);\n" +
    "    for (int i = 0; i < 4; ++i) {\n" +
    "        float w1 = lucQ[i];\n" +
    "        float delta0 = facX * (luc[i][0] - vc[0]);\n" +
    "        float delta1 = facY * (luc[i][1] - vc[1]);\n" +
    "        float d2 = delta0 * delta0 + delta1 * delta1;\n" +
    "        float extinction = exp(-d2 / gamma);\n" +
    "        vel[0] += extinction * delta1 * epsilon * w1;\n" +
    "        vel[1] += extinction * (delta0 * epsilon * w1);\n" +
    "\n" +
    "        w1 = lucL[i];\n" +
    "        delta0 = facX * (luc[i][2] - vc[0]);\n" +
    "        delta1 = facY * (luc[i][3] - vc[1]);\n" +
    "        d2 = delta0 * delta0 + delta1 * delta1;\n" +
    "        extinction = exp(-d2 / gamma);\n" +
    "        vel[0] += extinction * delta1 * epsilon * w1;\n" +
    "        vel[1] += extinction * (delta0 * epsilon * w1);\n" +
    "    \n" +
    "    }\n" +
    "    vel[0] += 0.05;\n" +
    "    vec2 velocity = mix(u_wind_min, u_wind_max, vel);\n" +
    "    return velocity;\n" +
    "}\n" +
    "\n" +
    "void main() {\n" +
    "    vec4 color = texture2D(u_particles, v_tex_pos);\n" +
    "    vec2 pos = vec2(\n" +
    "        color.r / 255.0 + color.b,\n" +
    "        color.g / 255.0 + color.a); // decode particle position from pixel RGBA\n\n" +
    "    vec2 velocity = mix(u_wind_min, u_wind_max, lookup_wind(pos));\n" +
    "    float speed_t = length(velocity) / length(u_wind_max);\n\n" +
    "    // take EPSG:4236 distortion into account for calculating where the particle moved\n" +
    "    float distortion = cos(radians(pos.y * 180.0 - 90.0)) + 0.1;\n" +
    // "    vec2 offset = vec2(velocity.x / 1.0, -velocity.y) * 0.0001 * u_speed_factor;\n\n" +
    "    vec2 offset = vec2(velocity.x * distortion, -velocity.y) * 0.0001 * u_speed_factor;\n\n" +
    // No distortion
    "    // update particle position, wrapping around the date line\n" +
    "    pos = fract(1.0 + pos + offset);\n\n" +
    "    // a random seed to use for the particle drop\n" +
    "    vec2 seed = (pos + v_tex_pos) * u_rand_seed;\n\n" +
    "    // drop rate is a chance a particle will restart at random position, to avoid degeneration\n" +
    "    float drop_rate = u_drop_rate + speed_t * u_drop_rate_bump;\n" +
    "    float drop = step(1.0 - drop_rate, rand(seed));\n\n" +
    "    vec2 random_pos = vec2(\n" +
    "        rand(seed + 1.3),\n" +
    "        rand(seed + 2.1));\n" +
    "    pos = mix(pos, random_pos, drop);\n\n" +
    // "    if (distance(mix(u_wind_min, u_wind_max, lookup_wind(pos)), vec2(0.0)) < 1.0) {\n" +
    // "        gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);" +
    // "    } else {\n" +
    "        // encode the new particle position back into RGBA\n" +
    "        gl_FragColor = vec4(\n" +
    "            fract(pos * 255.0),\n" +
    "            floor(pos * 255.0) / 255.0);\n" +
    // "    }\n" +
    "}\n";


var palette2 = {
    0.09: "#d73027",
    0.18: "#f46d43",
    0.27: "#f46d43",
    0.36: "#fdae61",
    0.45: "#fee090",
    0.54: "#ffffbf",
    0.63: "#e0f3f8",
    0.72: "#abd9e9",
    0.81: "#74add1",
    0.90: "#6694d1",
    1.00: "#4575b4"
};

// for (var vi = 0; vi < nbVortices; ++vi) {
//   var vp = vortices[vi];
//   // Distance to current vortex
//   var delta0 = vp[0] - xp;
//   var delta1 = vp[1] - yp;
//   var d2 = delta0 * delta0 + delta1 * delta1;
//   // Extinction factor
//   var extinction = Math.exp(-d2 / (vp[2] * gridScale));
//   mean[0] += extinction * delta1 * vp[3];
//   mean[1] += extinction * (-delta0 * vp[3]);
// }

var WindGL = function WindGL(gl) {
    this.gl = gl;
    this.fadeOpacity = 0.996; // how fast the particle trails fade on each frame
    this.speedFactor = 0.08; // 0.25; // how fast the particles move
    this.dropRate = 0.003; // how often the particles move to a random place
    this.dropRateBump = 0.01; // drop rate increase relative to individual particle speed
    this.eightVortices = [ // LUC
        0.3, 0.1,    0.1, 0.4,
        0.5, 0.5,    0.1, 0.7,
        0.7, 0.8,    0.3, 0.5,
        0.5, 0.2,    0.8, 0.3
    ];
    this.eightWeights1 = [
        2, 2, -2, 2
    ];
    this.eightWeights2 = [
        2, -2, -2, 2
    ];
    this.eightSpeeds = [
        0.0, 0.0,    0.0, 0.0,
        0.0, 0.0,    0.0, 0.0,
        0.0, 0.0,    0.0, 0.0,
        0.0, 0.0,    0.0, 0.0
    ];
    this.drawProgram = createProgram(gl, drawVert, drawFrag);
    this.screenProgram = createProgram(gl, quadVert, screenFrag);
    this.updateProgram = createProgram(gl, quadVert, updateFrag);
    this.quadBuffer = createBuffer(gl, new Float32Array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]));
    this.framebuffer = gl.createFramebuffer();
    this.setColorRamp(defaultRampColors);
    this.resize();
};

var prototypeAccessors = { numParticles: {} };

WindGL.prototype.resize = function resize () {
    var gl = this.gl;
    var emptyPixels = new Uint8Array(gl.canvas.width * gl.canvas.height * 4);
    // screen textures to hold the drawn screen for the previous and the current frame
    this.backgroundTexture = createTexture(gl, gl.NEAREST, emptyPixels, gl.canvas.width, gl.canvas.height);
    this.screenTexture = createTexture(gl, gl.NEAREST, emptyPixels, gl.canvas.width, gl.canvas.height);
};

WindGL.prototype.setColorRamp = function setColorRamp (colors) {
    // lookup texture for colorizing the particles according to their speed
    this.colorRampTexture = createTexture(this.gl, this.gl.LINEAR, getColorRamp(
        // colors
        palette2
    ), 16, 16);
};

prototypeAccessors.numParticles.set = function (numParticles) {
    var gl = this.gl;
    // we create a square texture where each pixel will hold a particle position encoded as RGBA
    var particleRes = this.particleStateResolution = Math.ceil(Math.sqrt(numParticles));
    this._numParticles = particleRes * particleRes;
    var particleState = new Uint8Array(this._numParticles * 4);
    for (var i = 0; i < particleState.length; i++) {
        particleState[i] = Math.floor(Math.random() * 256); // randomize the initial particle positions
    }
    // textures to hold the particle state for the current and the next frame
    this.particleStateTexture0 = createTexture(gl, gl.NEAREST, particleState, particleRes, particleRes);
    this.particleStateTexture1 = createTexture(gl, gl.NEAREST, particleState, particleRes, particleRes);

    var particleIndices = new Float32Array(this._numParticles);
    for (var i$1 = 0; i$1 < this._numParticles; i$1++) { particleIndices[i$1] = i$1; }
    this.particleIndexBuffer = createBuffer(gl, particleIndices);
};
prototypeAccessors.numParticles.get = function () {
    return this._numParticles;
};

// TODO cleanup / get this out to another project.
// WindGL.prototype.updateLuc = function() {
//   var v = this.eightVortices;
//   var s = this.eightSpeeds;
//   var w1 = this.eightWeights1;
//   var w2 = this.eightWeights2;
//   var w = [...w1, ...w2];
//   var mix = (min, max) => (x) => Math.max(min, Math.min(max, x));
//   var maxSpeed = 0.0005;
//
//   for (var i = 0; i < 8; ++i) {
//     var v1x = v[2 * i]; var v1y = v[2 * i + 1];
//     var s1x = s[2 * i]; var s1y = s[2 * i + 1];
//     var w1 = Math.abs(w[i]);
//     var sign1 = Math.sign(w[i]);
//     var ax = 0; var ay = 0;
//
//     for (var j = 0; j < 8; ++j) {
//       if (j === i) continue;
//       var v2x = v[2 * j]; var v2y = v[2 * j + 1];
//       // var s2x = s[2 * j]; var s2y = s[2 * j + 1];
//       var w2 = Math.abs(w[j]);
//       var sign2 = Math.sign(w[j]);
//
//       var d = Math.sqrt(Math.pow(v1x - v2x, 2) + Math.pow(v1y - v2y, 2));
//       d = Math.max(d, 0.001);
//       var G = 0.00000001;
//       var repulsion = -1.0; //sign1 === sign2 ? -1 : 1;
//       var f = repulsion * G * w1 * w2 / (d * d);
//       ax += f * (v2x - v1x) / d;
//       ay += f * (v2y - v1y) / d;
//     }
//
//     this.eightVortices[2 * i] = mix(0, 1)(v1x + s1x);
//     this.eightVortices[2 * i + 1] = mix(0, 1)(v1y + s1y);
//     this.eightSpeeds[2 * i] += mix(-maxSpeed, maxSpeed)(ax);
//     this.eightSpeeds[2 * i + 1] += mix(-maxSpeed, maxSpeed)(ay);
//
//     v1x = this.eightVortices[2 * i];
//     v1y = this.eightVortices[2 * i + 1];
//     //  var c = 10;
//     if (v1x <= 0 || v1x >= 1) this.eightSpeeds[2 * i] = -s1x;
//     if (v1y <= 0 || v1y >= 1) this.eightSpeeds[2 * i + 1] = -s1y;
//   }
// };

WindGL.prototype.setWind = function setWind (windData) {
    this.windData = windData;
    this.windTexture = createTexture(this.gl, this.gl.LINEAR, windData.image);
};

WindGL.prototype.draw = function draw () {
    var gl = this.gl;
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    bindTexture(gl, this.windTexture, 0);
    bindTexture(gl, this.particleStateTexture0, 1);
    bindTexture(gl, this.convTexture, 3);
    this.drawScreen();
    this.updateParticles();
};

WindGL.prototype.drawScreen = function drawScreen () {
    var gl = this.gl;
    // draw the screen into a temporary framebuffer to retain it as the background on the next frame
    bindFramebuffer(gl, this.framebuffer, this.screenTexture);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    this.drawTexture(this.backgroundTexture, this.fadeOpacity);
    this.drawParticles();
    bindFramebuffer(gl, null);
    // enable blending to support drawing on top of an existing background (e.g. a map)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    this.drawTexture(this.screenTexture, 1.0);
    gl.disable(gl.BLEND);
    // save the current screen as the background for the next frame
    var temp = this.backgroundTexture;
    this.backgroundTexture = this.screenTexture;
    this.screenTexture = temp;
};

WindGL.prototype.drawTexture = function drawTexture (texture, opacity) {
    var gl = this.gl;
    var program = this.screenProgram;
    gl.useProgram(program.program);
    bindAttribute(gl, this.quadBuffer, program.a_pos, 2);
    bindTexture(gl, texture, 2);
    gl.uniform1i(program.u_screen, 2);
    gl.uniform1f(program.u_opacity, opacity);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
};

WindGL.prototype.drawParticles = function drawParticles () {
    var gl = this.gl;
    var program = this.drawProgram;
    gl.useProgram(program.program);
    bindAttribute(gl, this.particleIndexBuffer, program.a_index, 1);
    bindTexture(gl, this.colorRampTexture, 2);
    gl.uniform1i(program.u_wind, 0);
    gl.uniform1i(program.u_particles, 1);
    gl.uniform1i(program.u_color_ramp, 2);
    gl.uniform1f(program.u_particles_res, this.particleStateResolution);
    gl.uniform2f(program.u_wind_res, this.windData.width, this.windData.height);
    gl.uniform2f(program.u_wind_min, this.windData.uMin, this.windData.vMin);
    gl.uniform2f(program.u_wind_max, this.windData.uMax, this.windData.vMax);
    gl.uniformMatrix4fv(program.luc, false, this.eightVortices);
    gl.uniform4fv(program.lucQ, this.eightWeights1);
    gl.uniform4fv(program.lucL, this.eightWeights2);
    gl.drawArrays(gl.POINTS, 0, this._numParticles);
};

WindGL.prototype.updateParticles = function updateParticles () {
    var gl = this.gl;
    bindFramebuffer(gl, this.framebuffer, this.particleStateTexture1);
    gl.viewport(0, 0, this.particleStateResolution, this.particleStateResolution);
    var program = this.updateProgram;
    gl.useProgram(program.program);
    bindAttribute(gl, this.quadBuffer, program.a_pos, 2);
    gl.uniform1i(program.u_wind, 0);
    gl.uniform1i(program.u_particles, 1);
    gl.uniform1f(program.u_rand_seed, Math.random());
    gl.uniform2f(program.u_wind_res, this.windData.width, this.windData.height);
    gl.uniform2f(program.u_wind_min, this.windData.uMin, this.windData.vMin);
    gl.uniform2f(program.u_wind_max, this.windData.uMax, this.windData.vMax);
    gl.uniform1f(program.u_speed_factor, this.speedFactor);
    gl.uniform1f(program.u_drop_rate, this.dropRate);
    gl.uniform1f(program.u_drop_rate_bump, this.dropRateBump);
    gl.uniformMatrix4fv(program.luc, false, this.eightVortices);
    gl.uniform4fv(program.lucQ, this.eightWeights1);
    gl.uniform4fv(program.lucL, this.eightWeights2);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    // swap the particle state textures so the new one becomes the current one
    var temp = this.particleStateTexture0;
    this.particleStateTexture0 = this.particleStateTexture1;
    this.particleStateTexture1 = temp;
};

Object.defineProperties( WindGL.prototype, prototypeAccessors );

function getColorRamp(colors) {
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 1;
    var gradient = ctx.createLinearGradient(0, 0, 256, 0);
    for (var stop in colors) {
        gradient.addColorStop(+stop, colors[stop]);
    }
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 256, 1);
    return new Uint8Array(ctx.getImageData(0, 0, 256, 1).data);
}


// TODO [REFACTOR] Make this a clean module.
var PARTICLE_LINE_WIDTH = 1;
var MAX_PARTICLE_AGE = 10000;
var FADE_FILL_STYLE = 'rgba(0, 0, 0, 0.97)';

// The palette can be easily tuned by adding colors.
var palette = [
    "#d73027",
    "#d73027",
    "#f46d43",
    "#f46d43",
    "#fdae61",
    "#fee090",
    "#ffffbf",
    "#e0f3f8",
    "#abd9e9",
    "#74add1",
    "#6694d1",
    "#4575b4"
];

// Draw objects
var af;
var buckets = [];
var NUMBER_BUCKETS = palette.length;
var particles = [];
var DOMElement;

// Simulation vars
var xPixels = 100;
var yPixels = 100;
var gridSize = xPixels * yPixels;
var gridScale = 100 * Math.sqrt(2);
var nbSamples;
var simulationType = 'gaussian';

// Simulation objects
var vortices = [];
var vortexSpeeds = [];
var vortexRanges = [];
var nbVortices = 100;
var MAX_VORTEX_NUMBER = 150;
var maxVectorFieldNorm = 5;

// Interaction objects
var isRightMouseDown = false;
var isLeftMouseDown = false;
var vortexAugmentationTimeout;
var REFRESH_RATE = 16; // 60 fps
var mouseRepulsionActive = false;
var mousePosition = [0, 0];

var g = null;

// var stamp1 = 0;
// var maxNumberParticles = 30000;
// var currentNumberParticles = 100;
var gl = null;
var windGL = null;
var heightVF = 180;
var widthVF = 360;
var staticVF = null;

var Windy =
    {
        start: function(
            gl, element, screenWidth, screenHeight, nbParticles, type)
        {
            this.end();
            DOMElement = element;
            // g = DOMElement.getContext("2d");

            // Internet Explorer

            windGL = new WindGL(gl);
            windGL.numParticles = nbParticles;
            staticVF = new Uint8Array(heightVF * widthVF);

            xPixels = screenWidth;
            yPixels = screenHeight;
            gridSize = xPixels * yPixels;
            gridScale = Math.sqrt(Math.pow(xPixels, 2) + Math.pow(yPixels, 2));
            nbSamples = nbParticles;
            // nbSamples = currentNumberParticles;
            // maxNumberParticles = nbParticles;

            if (type) simulationType = type;
            vortices = [];
            vortexSpeeds = [];
            vortexRanges = [];
            particles = [];
            buckets = [];

            this.makeVectorField();
            this.makeBuckets();
            this.makeParticles();
            this.animate();

            var windy = document.getElementById('windy');
            windy.addEventListener('contextmenu', function(e) {e.preventDefault()});
            windy.addEventListener('mousedown', this.mouseDownCallback.bind(this));
            windy.addEventListener('mouseup', this.mouseUpCallback.bind(this));
            windy.addEventListener('mousemove', this.mouseMoveCallback.bind(this));
            windy.addEventListener('mouseout', function() {mouseRepulsionActive = false}.bind(this));
        },

        end: function() {
            cancelAnimationFrame(af);
        },

        updateVF: function(domElement) {
            var c = 0;
            staticVF = new Uint8Array(heightVF * widthVF);
            for (var i = 0; i < heightVF; ++i) {
              var px = (i / heightVF) * xPixels;
              for (var j = 0; j < widthVF; ++j) {
                var py = (j / widthVF) * yPixels;
                var vec2 = this.computeVectorFieldAt(px, py);
                if (!vec2) vec2 = [0, 0];
                staticVF[c++] = vec2[0];
                staticVF[c++] = vec2[1];
              }
            }

            var fac = 4.0;
            var data = {
                width: DOMElement.offsetWidth ,
                height: DOMElement.offsetHeight ,
                uMin: 0, // -21.32,
                uMax: 30, // 26.8,
                vMin: 0, // -21.57,
                vMax: 30, // 21.42,
                image: staticVF
            };

            // if (!window.done) console.log(staticVF);
            // window.done = true;

            return data;
        },

        animate: function() {
            af = requestAnimationFrame(this.animate.bind(this));
            // var deltaT = Date.now() - stamp1;
            // var fps = 30;
            // var deltaThreshold = 1000 / fps;
            // if (deltaT < deltaThreshold) return;
            // stamp1 = Date.now();

            // this.update();
            // this.draw();

            if (!windGL.windData) {
                var wd = this.updateVF();
                windGL.windData = wd;
            }
            if (windGL.windData) {
                // windGL.updateLuc();
                windGL.draw();
            }
            // var stamp2 = ;
            // var deltaTime = Date.now() - stamp1;
            // if (deltaTime < 1000) {
            //   var numberOfParticlesToAdd = Math.min(100, maxNumberParticles - currentNumberParticles);
            //   if (numberOfParticlesToAdd > 0) {
            //     currentNumberParticles = currentNumberParticles + numberOfParticlesToAdd;
            //     nbSamples = currentNumberParticles;
            //     for (var i = 0; i < numberOfParticlesToAdd; ++i)
            //       particles.push(this.newParticle(i));
            //   }
            // } else {
            //   var numberOfParticlesToRemove = Math.min(0, currentNumberParticles - 100);
            //   if (numberOfParticlesToRemove > 0) {
            //     currentNumberParticles = currentNumberParticles - numberOfParticlesToRemove;
            //     nbSamples = currentNumberParticles;
            //     for (var i = 0; i < numberOfParticlesToRemove; ++i)
            //       particles.pop();
            //   }
            // }
        },

        makeBuckets: function() {
            // 1 bucket per color, NUMBER_BUCKETS colors.
            buckets = Array.apply(null, new Array(NUMBER_BUCKETS)).map(function(){return []});
                // Array.from(Array(NUMBER_BUCKETS).keys()).map(function(){return []});
        },

        addParticleToDrawBucket: function(particle, vector) {
            var maxVectorNorm = maxVectorFieldNorm;
            var thisVectorNorm = this.computeNorm(vector);
            var nbBuckets = buckets.length;

            var bucketIndex =
                thisVectorNorm < 0.001 ? 0 :
                    thisVectorNorm >= maxVectorNorm ? nbBuckets - 1 :
                        Math.ceil(nbBuckets * thisVectorNorm / maxVectorNorm);

            bucketIndex = bucketIndex >= buckets.length ? bucketIndex - 1 : bucketIndex;
            buckets[bucketIndex].push(particle);
        },

        makeParticles: function() {
            particles = [];
            for (var i = 0; i < nbSamples; ++i)
                particles.push(this.newParticle(i));
        },

        newParticle: function(particleRank) {
            var x0 = Math.floor(Math.random() * xPixels);
            var y0 = Math.floor(Math.random() * yPixels);
            return {
                x: x0,
                y: y0,
                xt: x0 + 0.01 * Math.random(),
                yt: y0 + 0.01 * Math.random(),
                age: Math.floor(Math.random() * MAX_PARTICLE_AGE),
                rank: particleRank
            };
        },

        evolveVectorField: function() {
            for (var vortex1Id = 0; vortex1Id < nbVortices; ++vortex1Id) {
                var vortex1 = vortices[vortex1Id];
                var o1 = vortex1[3] > 0; // orientation
                var mass1 = Math.abs(vortex1[3]);
                var charge1 = vortex1[2];
                var acceleration = [0, 0];
                // repulsion
                var coeff = 1 / gridScale; // 0.1;

                for (var vortex2Id = 0; vortex2Id < nbVortices; ++vortex2Id) {
                    if (vortex2Id === vortex1Id) continue;

                    var vortex2 = vortices[vortex2Id];
                    var o2 = vortex2[3] > 0;

                    var delta0 = coeff * (vortex1[0] - vortex2[0]);
                    var delta1 = coeff * (vortex1[1] - vortex2[1]);
                    var d2 =
                        Math.pow(delta0, 2) +
                        Math.pow(delta1, 2);

                    // Everything is repulsive
                    var sign = 1;
                    // Same sign vortices are attracted, opposite sign are repulsed
                    // o1 === o2 ? 1 : -1;

                    // !! Eulerian physics
                    // !! Charge could also be vortexI[3]
                    // !! Mass could also be vortexI[2]
                    var charge2 = vortex2[2];
                    var mass2 = Math.abs(vortex2[3]);
                    if (Math.abs(delta0) > 0.0001)
                        acceleration[0] += sign * Math.abs(charge1 * charge2 * mass1 * mass2) * delta0 /
                            (d2 * d2 * Math.abs(delta0));
                    if (Math.abs(delta1) > 0.0001)
                        acceleration[1] += sign * Math.abs(charge1 * charge2 * mass1 * mass2) * delta1 /
                            (d2 * d2 * Math.abs(delta1));
                }

                // Add four walls
                // coeff = 0.5;
                var v0x = coeff * vortex1[0]; var v0y = coeff * vortex1[1];
                var d0x = - coeff * xPixels + v0x; var d0y = - coeff * yPixels + v0y;
                var da = 0;
                if (Math.abs(v0x) > 0.001) {
                    da = (v0x) / (v0x * v0x * Math.abs(v0x));
                    acceleration[0] += da;
                    acceleration[1] += da * Math.sign(vortex1[3]);
                }
                if (Math.abs(d0x) > 0.001) {
                    da = (d0x) / (d0x * d0x * Math.abs(d0x));
                    acceleration[0] += da;
                    acceleration[1] += da * Math.sign(vortex1[3]);
                }
                if (Math.abs(v0y) > 0.001) {
                    da = (v0y) / (v0y * v0y * Math.abs(v0y));
                    acceleration[1] += da;
                    acceleration[0] -= da * Math.sign(vortex1[3]);
                }
                if (Math.abs(d0y) > 0.001) {
                    da = (d0y) / (d0y * d0y * Math.abs(d0y));
                    acceleration[1] += da;
                    acceleration[0] -= da * Math.sign(vortex1[3]);
                }

                // Add mouse
                if (mouseRepulsionActive) {
                    coeff *= 0.4;
                    var deltaX = coeff * (vortex1[0] - mousePosition[0]);
                    var deltaY = coeff * (vortex1[1] - mousePosition[1]);
                    var dist = deltaX * deltaX + deltaY * deltaY;
                    // Doesn't seem to matter after all...
                    if (Math.abs(deltaX) > 0.001) acceleration[0] += deltaX / (dist * dist * Math.abs(deltaX));
                    if (Math.abs(deltaY) > 0.001) acceleration[1] += deltaY / (dist * dist * Math.abs(deltaY));
                }

                var speedX = vortexSpeeds[vortex1Id][0] + 0.000001 * acceleration[0];
                var speedY = vortexSpeeds[vortex1Id][1] + 0.000001 * acceleration[1];

                vortexSpeeds[vortex1Id][0] = Math.sign(speedX) * Math.min(Math.abs(speedX), 0.3);
                vortexSpeeds[vortex1Id][1] = Math.sign(speedY) * Math.min(Math.abs(speedY), 0.3);

                var np0 = vortex1[0] + vortexSpeeds[vortex1Id][0];
                var np1 = vortex1[1] + vortexSpeeds[vortex1Id][1];
                vortex1[0] = Math.min(Math.max(np0, 0), xPixels);
                vortex1[1] = Math.min(Math.max(np1, 0), yPixels);

                // Update swiper.
                vortexRanges[vortex1Id] = this.computeVortexRange(vortex1);
            }
        },

        computeVortexRange: function(vortex) {
            var fadeCoefficient = 100;
            return [
                vortex[0] - vortex[2] * fadeCoefficient,
                vortex[0] + vortex[2] * fadeCoefficient,
                vortex[1] - vortex[2] * fadeCoefficient,
                vortex[1] + vortex[2] * fadeCoefficient
            ]
        },

        computeVectorFieldAt: function(xp, yp)
        {
            if (xp <= 1 || xp >= xPixels - 1 || yp <= 1 || yp >= yPixels - 1)
                return null;

            var mean = [0, 0];
            for (var vi = 0; vi < nbVortices; ++vi) {
                var vp = vortices[vi];
                var bounds = vortexRanges[vi];
                if (xp < bounds[0] || xp > bounds[1] || yp < bounds[2] || yp > bounds[3])
                    continue;

                // Distance to current vortex
                var delta0 = vp[0] - xp;
                var delta1 = vp[1] - yp;
                var d2 = delta0 * delta0 + delta1 * delta1;

                // To be clear with what we do here:
                // var gamma = vp[2] * gridScale;
                // var delta = [vp[0] - xp, vp[1] - yp, 0];
                // var up = [0, 0, vp[3]];

                // Cross product (the one used there)
                // var cross = [delta[1] * up[2], -delta[0] * up[2]];

                // Cute but odd (mangled cross product, interesting visual)
                // var cross = [delta[0] * up[2], -delta[1] * up[2]];

                var extinction = Math.exp(-d2 / (vp[2] * gridScale));
                mean[0] += extinction * delta1 * vp[3];    // cross[0];
                mean[1] += extinction * (-delta0 * vp[3]); // cross[1];
            }

            return mean;
        },

        makeVectorField: function() {
            vortices.length = 0;
            for (var v = 0; v < nbVortices; ++v) {
                var sg = Math.random() > 0.5 ? 1 : -1;
                var newVortex = [
                    Math.min(Math.random() * xPixels + 20, xPixels - 20), // x position
                    Math.min(Math.random() * yPixels + 20, yPixels - 20), // y position
                    5.0 * Math.max(0.25, Math.random()), // gaussian range
                    0.2 * sg * Math.max(Math.min(Math.random(), 0.5), 0.4) // gaussian intensity and clockwiseness
                ];

                vortices.push(newVortex);

                // Initial speeds
                vortexSpeeds.push([
                    0, // Math.random() - 0.5,
                    0  // Math.random() - 0.5
                ]);

                vortexRanges.push(this.computeVortexRange(newVortex));
            }
        },

        isNullVectorFieldAt: function(fx, fy)
        {
            return (fx <= 1 || fx >= xPixels - 1 || fy <= 1 || fy >= yPixels - 1);
        },

        computeNorm: function(vector) {
            return Math.sqrt(Math.pow(vector[0], 2) + Math.pow(vector[1], 2));
        },

        update: function() {
            // Empty buckets.
            for (var b = 0; b < buckets.length; ++b) buckets[b].length = 0;

            // Move particles and add them to buckets.
            for (var p = 0; p < particles.length; ++p) {
                var particle = particles[p];

                if (particle.age > MAX_PARTICLE_AGE) {
                    particles[particle.rank] = this.newParticle(particle.rank);
                }

                var x = particle.x;
                var y = particle.y;
                var v = this.computeVectorFieldAt(x, y);  // vector at current position

                if (v === null) {
                    // This particle is outside the grid
                    particle.age = MAX_PARTICLE_AGE;
                } else {
                    var xt = x + 0.1 * v[0];
                    var yt = y + 0.1 * v[1];

                    if (!this.isNullVectorFieldAt(xt, yt)) {
                        // The path of this particle is visible
                        particle.xt = xt;
                        particle.yt = yt;

                        if (Math.abs(x - xt) > 0.05 || Math.abs(y - yt) > 0.05) {
                            this.addParticleToDrawBucket(particle, v);
                        }
                    } else {
                        // This particle isn't visible, but still moves through the field.
                        particle.x = xt;
                        particle.y = yt;
                    }
                }

                particle.age += 1;
            }

            this.evolveVectorField();
        },

        // Enhancement: try out twojs
        // (Not fan of the loading overhead)
        draw: function() {
            g.lineWidth = PARTICLE_LINE_WIDTH;
            g.fillStyle = FADE_FILL_STYLE;
            g.mozImageSmoothingEnabled = false;
            g.webkitImageSmoothingEnabled = false;
            g.msImageSmoothingEnabled = false;
            g.imageSmoothingEnabled = false;

            // Fade existing particle trails.
            var prev = g.globalCompositeOperation;
            g.globalCompositeOperation = "destination-in";
            // g.fillStyle = "#ffffff";
            // g.fillStyle = "#000000";
            g.fillRect(0, 0, xPixels, yPixels);
            g.globalCompositeOperation = prev;

            // Draw new particle trails.
            var nbBuckets = buckets.length;
            for (var b = 0; b < nbBuckets; ++b) {
                var bucket = buckets[b];
                if (bucket.length > 0) {
                    g.beginPath();
                    g.strokeStyle = palette[b];
                    for (var p = 0; p < bucket.length; ++p) {
                        var particle = bucket[p];
                        var x = particle.x;
                        var xt = particle.xt;
                        var y = particle.y;
                        var yt = particle.yt;
                        // (This was for better extremal sampling:)
                        // g.moveTo(x - (xt - x) * 1.1, y - (yt - y) * 1.1);
                        // g.lineTo(xt + (xt - x) * 1.1, yt + (yt - y) * 1.1);
                        g.moveTo(x, y);
                        g.lineTo(xt, yt);
                        particle.x = xt;
                        particle.y = yt;
                    }
                    g.stroke();
                }
            }
        },

        getEventPositionInCanvas: function(event) {
            // YES, this is quick and dirty, please <i>please</i> be indulgent.
            // jQuery would have been a loading overhead
            // (Hyphenator is an overhead as well, but it is mandatory for Fr support).
            var windyElement = document.getElementById('windy');
            var rect = windyElement.getBoundingClientRect();
            var top = rect.top;
            var left = rect.left;
            return [event.clientX - left, event.clientY - top];
        },

        mouseDownCallback: function(event) {
            if (isLeftMouseDown) {
                // This should be possible with alt-tab, maybe.
                console.log('[MouseDownCallBack]: multiple mousedown events ' +
                    'without a mouseup.');
                return;
            }
            isLeftMouseDown = true;

            // Get coordinates for the click.
            var positionInCanvas = this.getEventPositionInCanvas(event);
            var sx = positionInCanvas[0];
            var sy = positionInCanvas[1];

            // Kind of a polyfill for detecting a right-click,
            // No jQuery should be involved.
            var rightclick =
                event.which ? (event.which === 3) :
                    event.button ? event.button === 2 : false;

            // We make it so the added vortex is always the last.
            var newVortex = [sx, sy, 1, rightclick ? -0.1 : 0.1];
            var newRange = this.computeVortexRange(newVortex);
            if (nbVortices < MAX_VORTEX_NUMBER) {
                nbVortices += 1;
            } else {
                vortices.shift();
                vortexRanges.shift();
                vortexSpeeds.shift();
            }
            vortices.push(newVortex);
            vortexRanges.push(newRange);
            vortexSpeeds.push([0, 0]);

            // Then we can progressively augment the size and speed of the created vortex.
            vortexAugmentationTimeout = setTimeout(
                this.augmentCreatedVortex.bind(this), REFRESH_RATE
            );
        },

        augmentCreatedVortex: function() {
            var lastVortexIndex = vortices.length - 1;
            var lastVortex = vortices[lastVortexIndex];

            if (mouseRepulsionActive) {
                lastVortex[0] = mousePosition[0];
                lastVortex[1] = mousePosition[1];
            }

            // Augment vortex.
            lastVortex[2] = Math.min(lastVortex[2] + 0.02, 5);
            if (lastVortex[3] > 0)
                lastVortex[3] = Math.min(lastVortex[3] + 0.01, 0.2);
            else
                lastVortex[3] = Math.max(lastVortex[3] - 0.01, -0.2);

            // Recompute vortex range.
            // Not strictly necessary: this is done at every vortex field evolution.
            vortexRanges[lastVortexIndex] = this.computeVortexRange(lastVortex);

            // Call again.
            vortexAugmentationTimeout = setTimeout(
                this.augmentCreatedVortex.bind(this), REFRESH_RATE
            );
        },

        mouseUpCallback: function(event) {
            mouseRepulsionActive = false;

            event.preventDefault();
            clearTimeout(vortexAugmentationTimeout);

            isLeftMouseDown = false;
        },

        mouseMoveCallback: function(event) {
            // Prevent dragging the canvas
            event.preventDefault();

            // Get new pointer position.
            var positionInCanvas = this.getEventPositionInCanvas(event);
            var sx = positionInCanvas[0];
            var sy = positionInCanvas[1];
            mousePosition = [sx, sy];

            // Check mouse status
            if (!isLeftMouseDown && !isRightMouseDown) {
                mouseRepulsionActive = true;
                return;
            }

            var lastVortexIndex = vortices.length - 1;
            var lastVortex = vortices[lastVortexIndex];

            var oldX = lastVortex[0];
            var oldY = lastVortex[1];
            var lastSpeed = vortexSpeeds[lastVortexIndex];
            var deltaX = sx - oldX;
            var deltaY = sy - oldY;

            lastSpeed[0] = Math.sign(deltaX) * Math.sqrt(Math.pow(deltaX / 500, 2));
            lastSpeed[1] = Math.sign(deltaY) * Math.sqrt(Math.pow(deltaY / 500, 2));

            lastVortex[0] = sx;
            lastVortex[1] = sy;
        }
    };

// 'Polyfill'
if (!window.requestAnimationFrame) {
    Windy = { start: function(){}, end: function(){} };
}
