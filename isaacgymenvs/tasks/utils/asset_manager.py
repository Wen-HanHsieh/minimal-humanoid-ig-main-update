import numpy as np
import os


class AssetManager:
    def __init__(self, asset_root, model_root_path: str) -> None:
        self.asset_root = asset_root
        self.model_root_path = model_root_path
        self.asset_files = self._scan()
        self.assets = []
        self.assets_dof_props = []
        self.rigid_body_counts = 0
        self.rigid_shape_counts = 0

        pass

    def __len__(self):
        # How many object instances do we have.
        return len(self.asset_files)

    def _scan(self):
        from pathlib import Path

        root_folder = Path(self.asset_root) / self.model_root_path
        asset_files = []
        for item in root_folder.iterdir():
            print("Iterating", item)
            if os.path.isdir(item) and os.path.exists(item / "model.urdf"):
                urdf_path = item / "model.urdf"
                urdf_relative_path = os.path.relpath(urdf_path, Path(self.asset_root))

                asset_files.append(urdf_relative_path)
                print(urdf_relative_path)
        return asset_files

    def load(self, env, asset_option, initializer):
        for f in self.asset_files:
            # print(f.as_posix())
            asset = env.gym.load_asset(env.sim, self.asset_root, f, asset_option)
            cube_dof_props = env.gym.get_asset_dof_properties(asset)
            initializer.initialize_object_dof(cube_dof_props)
            # print("CUBE_LR", cube_dof_props['lower'][0], cube_dof_props['upper'][0])
            self.assets.append(asset)
            self.assets_dof_props.append(cube_dof_props)

            self.rigid_body_counts += env.gym.get_asset_rigid_body_count(asset)
            self.rigid_shape_counts += env.gym.get_asset_rigid_shape_count(asset)

    def get_asset_rigid_body_count(self):
        return self.rigid_body_counts

    def get_asset_rigid_shape_count(self):
        return self.rigid_shape_counts

    def get_random_asset(self):
        i = np.random.randint(0, len(self.assets), 1)[0]
        return i, self.assets[i], self.assets_dof_props[i]

    def get_markers(self):
        # Marker handles
        if "toru_dec31" in self.model_root_path:
            cap_marker_handle_names = [
                "l10",
                "l11",
                "l12",
                "l13",
                "l10b",
                "l11b",
                "l12b",
                "l13b",
            ]
            base_marker_handle_names = [
                "l20",
                "l21",
                "l22",
                "l23",
                "l24",
                "l25",
                "l26",
                "l27",
            ]

        elif "m16" in self.model_root_path:
            cap_marker_handle_names = [
                "l10",
                "l11",
                "l12",
                "l13",
                "l10b",
                "l11b",
                "l12b",
                "l13b",
            ]
            base_marker_handle_names = [
                "l20",
                "l21",
                "l22",
                "l23",
                "l24",
                "l25",
                "l26",
                "l27",
            ]
            base_marker_handle_names += [
                "l30",
                "l31",
                "l32",
                "l33",
                "l34",
                "l35",
                "l36",
                "l37",
            ]
        else:
            cap_marker_handle_names = [
                "l10",
                "l11",
                "l12",
                "l13",
                "l10b",
                "l11b",
                "l12b",
                "l13b",
            ]
            base_marker_handle_names = ["l20", "l21", "l22", "l23"]

        print(cap_marker_handle_names, base_marker_handle_names)
        return cap_marker_handle_names, base_marker_handle_names
