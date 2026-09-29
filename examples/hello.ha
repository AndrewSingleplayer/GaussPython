// Run with:  python3 -m happ run examples/hello.ha     (or ./happ.sh run / happ.cmd run)

struct Splat { pos: vec3, color: vec4 }

fn brightness(s: Splat) -> f32 {
    return dot(s.color.rgb, vec3(0.299, 0.587, 0.114)) * s.color.a;
}

fn main() {
    let s = Splat(pos: vec3(0.0, 1.0, 2.0), color: vec4(1.0, 0.5, 0.25, 0.8));
    print("hello from HA++");
    print("splat at", s.pos, "brightness", brightness(s));
    var total = 0.0;
    for i in 0..1000 {
        total += sigmoid((i as f32) * 0.01 - 5.0);
    }
    print("sum of 1000 sigmoids:", total);
}
