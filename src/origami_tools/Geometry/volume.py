from .surface import *
from typing import Sequence
from stl import mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass    
class Volume():
    """ A class representing a volume defined by a list of surfaces. """
    surfaces : Sequence[Surface] 
    n_surfaces : int = 0

    def __post_init__(self):
        if len(self) < 1:
            raise ValueError("Volume must have at least one surface")
        for i in range(len(self)):
            if self[i].dimension != 3:
                raise ValueError("All surfaces must be in 3D")
        
        self.n_surfaces = len(self)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.surfaces[index]
        else:
            raise TypeError("Index must be an integer or a slice")
    
    def __len__(self):
        return len(self.surfaces)

    def copy(self):
        """Create a copy of the volume."""
        return Volume([point.copy() for point in self])
    

    def mesh_3D(self):
        """Create a 3D mesh representation of the volume."""
        
        surfaces = []
        for surface in self:
            surfaces.extend(surface.triangulate())
        volume_mesh = mesh.Mesh(np.zeros(len(surfaces), dtype=mesh.Mesh.dtype))
        for i, surface in enumerate(surfaces):
            for j in range(3):
                volume_mesh.vectors[i][j] = surface[j].as_array() # type: ignore
        
        return volume_mesh
    
    def __str__(self):
        """String representation of the volume."""
        pres = ""
        for surface in self:
            pres += str(surface) + "\n"
        return pres

    def wireframe(self):
        """Create a wireframe representation of the volume."""
        lines = []
        for surface in self:
            lines.extend(surface.edges())
        return lines
    
    def __add__(self, other):
        """add two volumes or surfaces together."""
        if not isinstance(other, (Volume, Surface)):
            raise ValueError("Other must be a Volume object")
        

        surfaces = []
        for surface in self:
            surfaces.append(surface.copy())
        if isinstance(other, Surface):
            surfaces.append(other.copy())
        else:
            for surface in other:
                surfaces.append(surface.copy())
        return Volume(surfaces)
    
    def show(self, save = False, path = None, alpha = 0.5, edgecolor = '0', facecolor = 'cyan'):
        ax = plt.figure().add_subplot(111, projection='3d')
        
        volume_mesh = self.mesh_3D()

        poly = Poly3DCollection(volume_mesh.vectors, alpha=alpha) # type: ignore
        poly.set_edgecolor(edgecolor)
        poly.set_facecolor(facecolor) # type: ignore
        
        # Auto scale to the mesh size
        scale = volume_mesh.points.flatten() # type: ignore
        scalex = []
        scaley = []
        scalez = []
        for i in range(int(len(scale) / 3)):
            scalex.append(scale[i * 3])
            scaley.append(scale[i * 3 + 1])
            scalez.append(scale[i * 3 + 2])
        ax.auto_scale_xyz(scalex, scaley, scalez) # type: ignore
        ax.add_collection3d(poly) # type: ignore
        ax.set_aspect('equal')

        ax.set_title('3D Volume')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') # type: ignore
        if save:
            if path is None:
                path = 'volume.png'
            plt.savefig(path)
            plt.close()
            # print("Volume saved to volume.png")
        else:
            plt.show() 

    def translate(self, v : Vec):
        """Translate the volume by a given vector."""
        if not isinstance(v, (Point, Vec)):
            raise ValueError("Translation vector must be a Point object")
        
        for i in range(len(self)):
            self[i].translate(v) 
    
    def save(self, filename : str):
        """Save the volume to a file."""
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string")
        if not filename.endswith('.stl'):
            filename += '.stl'
        
        volume_mesh = self.mesh_3D()
        volume_mesh.save(filename) # type: ignore
        print(f"Volume saved to {filename}")

@dataclass
class Parallelepiped(Volume):
    pass