import os
import shutil
import numpy as np
from trimesh.version import __version__ as trimesh_version
import trimesh as tm
import argparse


def export_urdf(
        coacd_path,
        output_directory,
        scale=0.1,
        color=[0.75, 0.75, 0.75],
        **kwargs):
    """
    Convert a Trimesh object into a URDF package for physics simulation.
    This breaks the mesh into convex pieces and writes them to the same
    directory as the .urdf file.

    Parameters
    ---------
    input_filename   : str
    output_directiry : str
                  The directory path for the URDF package

    Returns
    ---------
    mesh : Trimesh object
             Multi-body mesh containing convex decomposition
    """

    import lxml.etree as et
    # TODO: fix circular import
    from trimesh.exchange.export import export_mesh
    # Extract the save directory and the file name
    fullpath = os.path.abspath(output_directory)
    name = os.path.basename(fullpath)
    _, ext = os.path.splitext(name)

    if ext != '':
        raise ValueError('URDF path must be a directory!')

    # Create directory if needed
    if not os.path.exists(fullpath):
        os.mkdir(fullpath)
    elif not os.path.isdir(fullpath):
        raise ValueError('URDF path must be a directory!')

    # Perform a convex decomposition
    # if not exists:
    #     raise ValueError('No coacd available!')

    # argstring = f' -i {input_filename} -o {os.path.join(output_directory, "decomposed.obj")}'
    #
    # # pass through extra arguments from the input dictionary
    # for key, value in kwargs.items():
    #     argstring += ' -{} {}'.format(str(key),
    #                                   str(value))
    # os.system(coacd_path + argstring)
    #
    # convex_pieces = list(tm.load(os.path.join(
    #     output_directory, 'decomposed.obj'), process=False).split())
    #
    # # Get the effective density of the mesh
    # mesh = tm.load(input_filename,force="mesh", process=False)
    # effective_density = mesh.volume / sum([
    #     m.volume for m in convex_pieces])
    convex_pieces_name = [os.path.splitext(piece)[0] for piece in os.listdir(coacd_path) if
                     'coacd_convex_piece' in piece]
    convex_pieces = [tm.load(os.path.join(coacd_path,piece)) for piece in os.listdir(coacd_path) if 'coacd_convex_piece' in piece]
    peices_mesh_z_max = [np.max(mesh.vertices[:, -1]) for mesh in convex_pieces]

    # get top piece
    top_piece_name = convex_pieces_name[np.argmax(peices_mesh_z_max)]
    top_piece = convex_pieces[np.argmax(peices_mesh_z_max)]
    convex_pieces.remove(top_piece)
    convex_pieces = convex_pieces + [top_piece]
    convex_pieces_name.remove(top_piece_name)
    convex_pieces_name = convex_pieces_name + [top_piece_name]
    cap_name = top_piece_name

    # open an XML tree
    root = et.Element('robot', name='root')
    # add bottle body inetial
    link_body = et.SubElement(root, 'link', name='bottle_body')
    inertial_body = et.SubElement(link_body, 'inertial')
    et.SubElement(inertial_body, 'origin', xyz="0 0 0", rpy="0 0 0")
    et.SubElement(link_body, 'mass', value='{:.2E}'.format(0.01))
    # add bottle cap inetial
    link_cap = et.SubElement(root, 'link', name='bottle_cap')
    et.SubElement(link_cap, 'origin', xyz="0 0 0.002", rpy="0 0 0")
    et.SubElement(link_cap, 'mass', value='{:.2E}'.format(0.01))
    # Loop through all pieces, adding each as a link
    prev_link_name = None
    i = 0
    for piece in convex_pieces:
        # Save each nearly convex mesh out to a file
        # piece_name = '{}-coacd_convex_piece_{}'.format(name, i)
        piece_name = convex_pieces_name[i]
        piece_filename = '{}.obj'.format(piece_name)
        piece_filepath = os.path.join(fullpath, piece_filename)
        # export_mesh(piece, piece_filepath)

        # # Set the mass properties of the piece
        # piece.center_mass = mesh.center_mass
        # piece.density = effective_density * mesh.density
        # piece.center_mass = 0.01
        if piece_name != cap_name:

            geom_name = '{}'.format(piece_filename)
            visual = et.SubElement(link_body, 'visual')
            et.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
            geometry = et.SubElement(visual, 'geometry')

            et.SubElement(geometry, 'mesh', filename=geom_name,
                          scale="{:.4E} {:.4E} {:.4E}".format(scale,
                                                              scale,
                                                              scale))
            material = et.SubElement(visual, 'material', name='')
            color = [0.6, 0.2, 0.2]
            et.SubElement(material,
                          'color',
                          rgba="{:.2E} {:.2E} {:.2E} 1".format(color[0],
                                                               color[1],
                                                               color[2]))
            # Collision Information
            collision = et.SubElement(link_body, 'collision')
            et.SubElement(collision, 'origin', xyz="0 0 0", rpy="0 0 0")
            geometry = et.SubElement(collision, 'geometry')
            et.SubElement(geometry, 'mesh', filename=geom_name,
                          scale="{:.4E} {:.4E} {:.4E}".format(scale,
                                                              scale,
                                                              scale))
        else:
            geom_name = '{}'.format(piece_filename)
            visual = et.SubElement(link_cap, 'visual')
            et.SubElement(visual, 'origin', xyz="0 0 0.002", rpy="0 0 0")
            geometry = et.SubElement(visual, 'geometry')

            et.SubElement(geometry, 'mesh', filename=geom_name,
                          scale="{:.4E} {:.4E} {:.4E}".format(scale,
                                                              scale,
                                                              scale))
            material = et.SubElement(visual, 'material', name='')
            color = [0.3, 0.6, 0.6]
            et.SubElement(material,
                          'color',
                          rgba="{:.2E} {:.2E} {:.2E} 1".format(color[0],
                                                               color[1],
                                                               color[2]))

            # Collision Information
            collision = et.SubElement(link_cap, 'collision')
            et.SubElement(collision, 'origin', xyz="0 0 0.002", rpy="0 0 0")
            geometry = et.SubElement(collision, 'geometry')
            et.SubElement(geometry, 'mesh', filename=geom_name,
                          scale="{:.4E} {:.4E} {:.4E}".format(scale,
                                                              scale,
                                                              scale))
        i += 1
    joint_name = 'bottle_cap_joint'
    joint = et.SubElement(root,
                          'joint',
                          name=joint_name,
                          type='revolute')
    et.SubElement(joint, 'origin', xyz="0 0 0", rpy="0 0 0")
    et.SubElement(joint, 'parent', link='bottle_body')
    et.SubElement(joint, 'child', link='bottle_cap')
    et.SubElement(joint, 'axis', xyz="0 0 1")
    et.SubElement(joint, 'limit', lower="0.00", upper="6.28")
    et.SubElement(joint, 'dynamics', damping="0.05")




    # Write URDF file
    tree = et.ElementTree(root)
    urdf_filename = '{}.urdf'.format(name)
    tree.write(os.path.join(fullpath, urdf_filename),
               pretty_print=True)

    # # Write Gazebo config file
    # root = et.Element('model')
    # model = et.SubElement(root, 'name')
    # model.text = name
    # version = et.SubElement(root, 'version')
    # version.text = '1.0'
    # sdf = et.SubElement(root, 'sdf', version='1.4')
    # sdf.text = '{}.urdf'.format(name)
    #
    # author = et.SubElement(root, 'author')
    # et.SubElement(author, 'name').text = 'trimesh {}'.format(trimesh_version)
    # et.SubElement(author, 'email').text = 'blank@blank.blank'
    #
    # description = et.SubElement(root, 'description')
    # description.text = name
    #
    # tree = et.ElementTree(root)
    # tree.write(os.path.join(fullpath, 'model.config'))

    return np.sum(convex_pieces)


# def decompose(result_path, object_code):
#
#     print(f'decomposition: {object_code}')
#
#     if os.path.exists(os.path.join(result_path, object_code, 'coacd')):
#         shutil.rmtree(os.path.join(result_path, object_code, 'coacd'))
#     os.makedirs(os.path.join(args.result_path, object_code, 'coacd'))
#     coacd_params = {
#         't': args.t,
#         'k': args.k
#     }
#     export_urdf(args.coacd_path, os.path.join(args.data_root_path, object_code + ".obj"),
#                 os.path.join(args.result_path, object_code, 'coacd'), **coacd_params)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--k', default=0.3, type=float)
#     parser.add_argument('--t', default=0.08, type=float)
#
#     parser.add_argument('--coacd_path', required=True, type=str)
#     parser.add_argument('--result_path', required=True, type=str)
#     parser.add_argument('--data_root_path', required=True, type=str)
#     parser.add_argument('--object_code', required=True, type=str)
#
#     args = parser.parse_args()
#
#     # check whether arguments are valid and process arguments
#
#     if not os.path.exists(args.result_path):
#         os.makedirs(args.result_path)
#
#     if not os.path.exists(args.data_root_path):
#         raise ValueError(
#             f'data_root_path {args.data_root_path} doesn\'t exist')
#
#     decompose(args, args.object_code)
