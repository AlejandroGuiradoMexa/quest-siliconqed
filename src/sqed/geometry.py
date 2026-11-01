"""Geometries (netgen OCC) for the 2D models: a dielectric cylinder and a silicon nanobeam cavity
with an optional bowtie at its centre. Lengths in um. Regions: 'si' (silicon), 'air', 'pml',
'src' (a small disc at the emitter position), outer boundary 'outer'."""
import numpy as np
from netgen.occ import Circle, Rectangle, MoveTo, Glue, OCCGeometry, X, Y
from ngsolve import Mesh, pml


def _name(face, name, maxh=None):
    face.faces.name = name
    if maxh:
        face.faces.maxh = maxh
    return face


def cylinder(R=0.5, Rd=2.0, Rp=4.0, h_in=0.04, h_out=0.1, src=None, src_r=0.002, curve=4):
    """Dielectric cylinder radius R in air, circular PML between Rd and Rp. src = (x, y) adds a source disc."""
    cyl = _name(Circle((0, 0), R).Face(), "si", h_in)
    parts = []
    if src is not None:
        s = _name(Circle(src, src_r).Face(), "src", src_r / 2)
        inside = np.hypot(*src) < R
        if inside:
            cyl = cyl - s
            cyl.faces.name = "si"; cyl.faces.maxh = h_in
        parts.append(s)
    air = Circle((0, 0), Rd).Face() - Circle((0, 0), R).Face()
    for p in parts:
        if np.hypot(*src) >= R:
            air = air - p
    _name(air, "air", h_out)
    outer = Circle((0, 0), Rp).Face()
    outer.edges.name = "outer"
    pm = _name(outer - Circle((0, 0), Rd).Face(), "pml", h_out * 1.3)
    shape = Glue([cyl, air, pm] + parts)
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=h_out * 1.3, grading=0.25))
    mesh.Curve(curve)
    mesh.SetPML(pml.Radial(rad=Rd, alpha=2.0j, origin=(0, 0)), "pml")
    return mesh


# Nanobeam of Kim et al., Nano Lett. 25, 4347 (2025), Supporting Information S1 (220 nm SOI, G centre at 1277 nm):
# lattice constant 350 nm, hole radius 115 nm, four-hole taper with radii 107.5 -> 85 nm and periods 325 -> 250 nm
# (outer -> inner), beam width 520 nm. Here symmetric, with n_mirror holes on each side.
KIM = dict(a=0.350, r=0.115, taper_r=(0.1075, 0.100, 0.0925, 0.085), taper_a=(0.325, 0.300, 0.275, 0.250), W=0.520)


def nanobeam_holes(scale=1.0, n_mirror=8, p=KIM):
    """Centres and radii of the holes for x > 0 (the beam is mirror-symmetric about x = 0)."""
    ta, tr = [scale * v for v in p["taper_a"]], [scale * v for v in p["taper_r"]]
    xs = [ta[-1] / 2]
    rs = [tr[-1]]
    # outward from the centre: spacings a4, a3, a2, a1 then the mirror period
    for i in range(len(ta) - 2, -1, -1):
        xs.append(xs[-1] + ta[i])
        rs.append(tr[i])
    for _ in range(n_mirror):
        xs.append(xs[-1] + scale * p["a"])
        rs.append(scale * p["r"])
    return np.array(xs), np.array(rs)


def nanobeam(scale=1.0, n_mirror=8, bowtie=None, src_r=0.002, margin=0.8, tpml=0.8, h_si=0.06, h_air=0.14,
             h_bridge=None, p=KIM, perturb=None, curve=3):
    """2D (top view) nanobeam cavity. bowtie = dict(w=bridge width, hb=base height, xb=base position) replaces the
    innermost hole pair by two air trapezoids pointing at the centre, leaving a silicon bridge of width w.
    perturb = dict(dr=array, dx=array) adds per-hole radius and position changes (for both sides, length 2*nholes:
    first the x > 0 holes, then the x < 0 holes). A source disc of radius src_r sits at the centre."""
    xs, rs = nanobeam_holes(scale, n_mirror, p)
    W = scale * p["W"]
    Xd = xs[-1] + rs[-1] + margin
    Yd = W / 2 + margin
    beam = Rectangle(2 * (Xd + tpml), W).Face().Move((-(Xd + tpml), -W / 2, 0))
    holes = []
    nh = len(xs)
    for side, sgn in enumerate((1, -1)):
        for i, (x, r) in enumerate(zip(xs, rs)):
            if bowtie is not None and i == 0:
                continue
            dr = perturb["dr"][side * nh + i] if perturb is not None else 0.0
            ddx = perturb["dx"][side * nh + i] if perturb is not None else 0.0
            holes.append(Circle((sgn * x + ddx, 0), r + dr).Face())
    if bowtie is not None:
        w, hb, xb = bowtie["w"], scale * bowtie.get("hb", 2 * p["taper_r"][-1]), scale * bowtie.get("xb", 0.210)
        t = bowtie.get("t", w)
        for sgn in (1, -1):
            pts = [(sgn * w / 2, -t / 2), (sgn * xb, -hb / 2), (sgn * xb, hb / 2), (sgn * w / 2, t / 2)]
            if sgn < 0:
                pts = pts[::-1]  # counter-clockwise: a clockwise face hangs the netgen mesher
            wp = MoveTo(*pts[0])
            for q in pts[1:]:
                wp = wp.LineTo(*q)
            holes.append(wp.Close().Face())
    si = beam
    for h in holes:
        si = si - h
    src = _name(Circle((0, 0), src_r).Face(), "src", src_r / 2)
    si = si - src
    # refinement box around the centre
    hb_ = h_bridge or (bowtie["w"] / 3 if bowtie else h_si / 2)
    si.faces.name = "si"
    si.faces.maxh = h_si
    box = Rectangle(2 * Xd, 2 * Yd).Face().Move((-Xd, -Yd, 0))
    outer = Rectangle(2 * (Xd + tpml), 2 * (Yd + tpml)).Face().Move((-(Xd + tpml), -(Yd + tpml), 0))
    outer.edges.name = "outer"
    air = _name(box - beam, "air", h_air)
    pm = outer - box
    si_in = si * box
    si_in.faces.name = "si"; si_in.faces.maxh = h_si
    si_pml = si * pm
    si_pml.faces.name = "pml"; si_pml.faces.maxh = h_air
    air_pml = _name(pm - beam, "pml", h_air)
    holes_in = []
    for h in holes:
        hh = h * box
        hh.faces.name = "air"; hh.faces.maxh = h_si
        holes_in.append(hh)
    shape = Glue([si_in, si_pml, air, air_pml, src] + holes_in)
    if bowtie is not None:
        # local refinement near the bridge
        shape.faces[0].maxh = h_si
        for v in shape.vertices:
            if abs(v.p.x) < 0.05 * scale + bowtie["w"] and abs(v.p.y) < 0.05 * scale + bowtie["w"]:
                v.maxh = hb_
    geo = OCCGeometry(shape, dim=2)
    mesh = Mesh(geo.GenerateMesh(maxh=h_air, grading=0.2))
    mesh.Curve(curve)
    mesh.SetPML(pml.Cartesian(mins=(-Xd, -Yd), maxs=(Xd, Yd), alpha=2.0j), "pml")
    return mesh, dict(Xd=Xd, Yd=Yd, W=W, xs=xs, rs=rs)
