// HoloPixel Web プレイヤー（WebGL2）
// .hpx の mobile プロファイル（mobile/hpx.mp4 または mobile/still.webp）を読み、
// 視点の 3D 位置 [m]（窓座標）から見た 1 枚を描く。計算は td/hpx_player.glsl と同じ。
// 1 フレームは [カラー | 深度 | アルファ] を横に並べたもの（docs/SPEC_v0.md 9 章）。

const VS = `#version 300 es
in vec2 aPos; out vec2 vUV;
void main(){ vUV = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 fragColor;
uniform sampler2D uTex;
uniform vec3 uCam;     // 視点の位置（窓座標系）[m]
uniform vec4 uWin;     // 窓の中心 xy, 幅・高さ [m]
uniform vec4 uRef;     // 基準カメラ位置 xyz, tan(fovY/2)
uniform vec4 uAtlas;   // cols, rows, レイヤー数, タイルのアスペクト比
uniform vec4 uMarch;   // ステップ数, near, far, 厚み判定
uniform vec2 uBins[8]; // 各レイヤーの (zNear, zFar)
uniform vec2 uTexel;   // アトラス 1 枚の 1 画素の大きさ（0〜1）
uniform vec3 uBg;

// タイル内 uv → テクスチャ座標。region 0 = カラー, 1 = 深度, 2 = アルファ
vec2 tileUV(int i, vec2 uv, float region){
	vec2 grid = uAtlas.xy;
	uv = clamp(uv, 0.5 * uTexel * grid, 1.0 - 0.5 * uTexel * grid);
	float c = mod(float(i), grid.x), r = floor(float(i) / grid.x);
	vec2 a = vec2((c + uv.x) / grid.x, 1.0 - (r + 1.0 - uv.y) / grid.y);
	return vec2((region + a.x) / 3.0, a.y);
}
vec3 toRef(vec3 p){
	vec3 q = p - uRef.xyz;
	return vec3(q.xy / q.z / (uRef.w * vec2(uAtlas.w, 1.0)) * 0.5 + 0.5, q.z);
}
bool inside(vec2 uv){ return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0))); }
float layerZ(int i, vec2 uv){
	float d = texture(uTex, tileUV(i, uv, 1.0)).g;   // レイヤーごとの正規化逆深度
	vec2 b = uBins[i];
	return 1.0 / mix(1.0 / b.y, 1.0 / b.x, d);
}
void main(){
	vec3 W = vec3(uWin.xy + (vUV - 0.5) * uWin.zw, 0.0);
	vec3 dir = W - uCam;
	vec4 acc = vec4(0.0);
	int K = int(uAtlas.z), N = int(uMarch.x);
	if (dir.z > 1e-4) {
		for (int i = 0; i < 8; i++) {
			if (i >= K || acc.a >= 0.995) break;
			bool last = (i == K - 1);
			vec2 bin = uBins[i];
			float z0 = max(bin.x * 0.9, uMarch.y);
			float z1 = last ? uMarch.z * 1.01 : bin.y * 1.1;
			float prevS = -1.0, hitS = -1.0;
			for (int n = 0; n <= N; n++) {
				float z = 1.0 / mix(1.0 / z0, 1.0 / z1, float(n) / float(N));
				float s = (z + uRef.z - uCam.z) / dir.z;
				if (s <= 0.0) continue;
				vec3 r = toRef(uCam + s * dir);
				if (!inside(r.xy)) { prevS = -1.0; continue; }
				if (r.z >= layerZ(i, r.xy)) {
					float a = prevS > 0.0 ? prevS : s, b = s;
					for (int k = 0; k < 6; k++) {
						float m = 0.5 * (a + b);
						vec3 rm = toRef(uCam + m * dir);
						if (rm.z >= layerZ(i, rm.xy)) b = m; else a = m;
					}
					vec3 rh = toRef(uCam + b * dir);
					float zh = layerZ(i, rh.xy);
					if (last || rh.z - zh < zh * uMarch.w) hitS = b;
					break;
				}
				prevS = s;
			}
			if (hitS > 0.0) {
				vec3 rh = toRef(uCam + hitS * dir);
				vec3 col = texture(uTex, tileUV(i, rh.xy, 0.0)).rgb;
				float al = texture(uTex, tileUV(i, rh.xy, 2.0)).g;
				acc.rgb += (1.0 - acc.a) * al * col;
				acc.a += (1.0 - acc.a) * al;
			}
		}
	}
	fragColor = vec4(acc.rgb + (1.0 - acc.a) * uBg, 1.0);
}`;

export class HoloPixelPlayer {
	constructor(canvas) {
		this.canvas = canvas;
		const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
		if (!gl) throw new Error('WebGL2 が使えません');
		this.gl = gl;
		this.prog = this._program(VS, FS);
		const buf = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, buf);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
		const loc = gl.getAttribLocation(this.prog, 'aPos');
		gl.enableVertexAttribArray(loc);
		gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
		this.tex = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, this.tex);
		for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
			[gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
		this.u = {};
		for (const n of ['uTex', 'uCam', 'uWin', 'uRef', 'uAtlas', 'uMarch', 'uBins', 'uTexel', 'uBg'])
			this.u[n] = gl.getUniformLocation(this.prog, n);
		this.cam = [0, 0, -3];
		this.steps = 64;
		this.thickness = 0.15;
		this.ready = false;
	}

	_program(vs, fs) {
		const gl = this.gl;
		const sh = (type, src) => {
			const s = gl.createShader(type);
			gl.shaderSource(s, src); gl.compileShader(s);
			if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
			return s;
		};
		const p = gl.createProgram();
		gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
		gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
		gl.linkProgram(p);
		if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
		return p;
	}

	// .hpx フォルダの URL を読む。preferStill = true なら静止画（still.webp）を使う
	async load(base, preferStill = false) {
		base = base.replace(/\/$/, '');
		const m = await (await fetch(base + '/manifest.json')).json();
		if (!m.mobile) throw new Error('mobile プロファイルがありません（tools/hpx_pack.py --profiles mobile）');
		this.m = m;
		if (this.video) { this.video.pause(); this.video.removeAttribute('src'); this.video.load(); }
		this.video = null;
		this.source = await (preferStill ? this._image(base + '/' + m.mobile.still) : this._video(base + '/' + m.mobile.video)
			.catch(() => this._image(base + '/' + m.mobile.still)));
		this.isVideo = this.source instanceof HTMLVideoElement;
		if (this.isVideo) this.video = this.source;
		this.cam = m.refCamera.position.slice();
		this.ready = true;
		this._upload();
		return m;
	}

	_video(url) {
		return new Promise((resolve, reject) => {
			const v = document.createElement('video');
			Object.assign(v, { muted: true, loop: true, playsInline: true, crossOrigin: 'anonymous', preload: 'auto' });
			v.setAttribute('playsinline', '');
			v.src = url;
			v.addEventListener('loadeddata', () => resolve(v), { once: true });
			v.addEventListener('error', () => reject(new Error('動画を読めません: ' + url)), { once: true });
			v.load();
		});
	}

	_image(url) {
		return new Promise((resolve, reject) => {
			const im = new Image();
			im.crossOrigin = 'anonymous';
			im.onload = () => resolve(im);
			im.onerror = () => reject(new Error('画像を読めません: ' + url));
			im.src = url;
		});
	}

	_upload() {
		const gl = this.gl;
		gl.bindTexture(gl.TEXTURE_2D, this.tex);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);   // TD と同じく v = 0 を画像の下端にする
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.source);
	}

	// 出力の縦横比（窓の縦横比）
	get aspect() { const s = this.m.window.size; return s[0] / s[1]; }

	render() {
		if (!this.ready) return;
		const gl = this.gl, m = this.m, at = m.layers.atlas, mob = m.mobile.atlas;
		if (this.isVideo && this.video.readyState >= 2) this._upload();
		gl.viewport(0, 0, this.canvas.width, this.canvas.height);
		gl.useProgram(this.prog);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this.tex);
		gl.uniform1i(this.u.uTex, 0);
		const rc = m.refCamera, w = m.window;
		gl.uniform3fv(this.u.uCam, this.cam);
		gl.uniform4f(this.u.uWin, w.center[0], w.center[1], w.size[0], w.size[1]);
		gl.uniform4f(this.u.uRef, ...rc.position, Math.tan(rc.fovY * Math.PI / 360));
		gl.uniform4f(this.u.uAtlas, at.cols, at.rows, m.layers.count, at.tileWidth / at.tileHeight);
		gl.uniform4f(this.u.uMarch, this.steps, m.depth.near, m.depth.far, this.thickness);
		const bins = new Float32Array(16);
		m.layers.bins.forEach((b, i) => { bins[2 * i] = b[0]; bins[2 * i + 1] = b[1]; });
		gl.uniform2fv(this.u.uBins, bins);
		gl.uniform2f(this.u.uTexel, 1 / mob.width, 1 / mob.height);
		gl.uniform3f(this.u.uBg, 0, 0, 0);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
	}
}
