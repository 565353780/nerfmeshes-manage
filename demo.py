#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nerfmeshes_manage.Demo.train import demo as demo_train
from nerfmeshes_manage.Demo.mesh import demo as demo_nerf_mesh
#  from nerfmeshes_manage.Demo.mesh_surface_ray import demo as demo_mesh_surface_ray

if __name__ == "__main__":
    #  demo_train()
    demo_nerf_mesh()
    #  demo_mesh_surface_ray()
