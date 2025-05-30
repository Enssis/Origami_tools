__name__ = "origami_tools.test.test1"
from ..geometry import *
from ..decoupe_laser import *


p1 = Point(1, 2)
p2 = Point(3, 4)
p3 = Point(5, 6)

poly = Polygon([p1, p2, p3])

def test_add():
    assert p1 + p2 == Point(4, 6)
    assert p1 + p2 + p3 == Point(9, 12)

def test_patron():
    patron = Patron()
    assert patron is not None
    assert patron.shapes == []
    patron.add_shapes([poly])
    assert len(patron.shapes) == 1
    assert type(patron.shapes[0]) == DrawnShapes
    patron.save_PDF(__file__.replace("test1.py", ""))
    patron.save_SVG(__file__.replace("test1.py", ""))


test_add()
test_patron()
print("All tests passed!")