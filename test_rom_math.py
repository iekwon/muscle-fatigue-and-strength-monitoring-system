import math


def q_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    return (w/n, x/n, y/n, z/n)


def q_conj(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b

    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    )


def quaternion_from_x_rotation(degrees):
    r = math.radians(degrees) / 2.0
    return (math.cos(r), math.sin(r), 0.0, 0.0)


def relative_angle_degrees(q1, q2):
    q_rel = q_mul(q_conj(q1), q2)
    q_rel = q_normalize(q_rel)

    w, x, y, z = q_rel

    angle = 2.0 * math.atan2(
        math.sqrt(x*x + y*y + z*z),
        abs(w)
    )

    return math.degrees(angle)


q1 = quaternion_from_x_rotation(0)

for expected in [0, 30, 60, 90, 120, 150]:
    q2 = quaternion_from_x_rotation(expected)

    measured = relative_angle_degrees(q1, q2)

    print(
        f"Expected: {expected:3d} deg   "
        f"Measured: {measured:6.2f} deg"
    )
