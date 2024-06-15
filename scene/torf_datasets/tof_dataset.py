import numpy as np
import os
import scipy
import scipy.io

from scene.torf_datasets.utils.utils import normalize_im_max, depth_from_tof
from scene.torf_datasets.utils.camera_utils import crop_mvs_input, scale_mvs_input, get_extrinsics, get_camera_params, recenter_poses
from scene.torf_datasets.utils.sampling_utils import *

MAX_ITERS = 100000

# We work on ToF images
class ToFDataset(object):
    def __init__(
        self,
        args,
        file_endings={
            'tof': 'mat',
            'color': 'mat',
            'depth': 'mat',
            'motion_mask': 'npy',
            'cams': 'npy',
            },
        ):
        self.args = args
        self.file_endings = file_endings

        self._build_data_list(args)
        self._build_dataset(args)
        self._create_splits(args)

    def _build_data_list(self, args):
        # Create idx map
        self.get_frame_id = {} # Frame ID
        self.get_view_id = {} # View ID
        frame_id = 0
        view_id = 0

        while view_id < args.total_num_views and frame_id < MAX_ITERS:
            tof_filename = self._get_tof_filename(frame_id)
            color_filename = self._get_color_filename(frame_id)
            depth_filename = self._get_depth_filename(frame_id)
            motion_mask_filename = self._get_motion_mask_filename(frame_id)

            if os.path.isfile(tof_filename) or \
                os.path.isfile(color_filename) or \
                os.path.isfile(depth_filename) or \
                os.path.isfile(motion_mask_filename):
                self.get_frame_id[view_id] = frame_id
                self.get_view_id[frame_id] = view_id
                view_id += 1
            
            frame_id += 1

        # Create data list
        self.data_list = list( range(args.total_num_views) )

    def _get_image_name(self, frame_id):
        return f'{frame_id:04d}'
    
    def _get_tof_filename(self, frame_id):
        tof_filename = os.path.join(
                self.args.source_path,
                f'tof/{frame_id:04d}.{self.file_endings["tof"]}' 
                )
        
        return tof_filename

    def _get_color_filename(self, frame_id):
        color_filename = os.path.join(
                self.args.source_path,
                f'color/{frame_id:04d}.{self.file_endings["color"]}' 
                )
        
        return color_filename

    def _get_depth_filename(self, frame_id):
        depth_filename = os.path.join(
                self.args.source_path,
                f'depth/{frame_id:04d}.{self.file_endings["depth"]}' 
                )
        
        return depth_filename

    def _get_motion_mask_filename(self, frame_id):
        motion_mask_filename = os.path.join(
                self.args.source_path,
                f'motion_mask/{frame_id:04d}.{self.file_endings["motion_mask"]}' 
                )
        
        return motion_mask_filename

    def _build_dataset(self, args):
        # Empty dataset
        self.dataset = {
            'tof_intrinsics': [],
            'tof_extrinsics': [],
            'tof_poses': [],
            'tof_light_poses': [],
            'tof_images': [],

            'color_intrinsics': [],
            'color_extrinsics': [],
            'color_poses': [],
            'color_light_poses': [],
            'color_images': [],

            'depth_images': [],
            'motion_masks': [],
            'bounds': [],
        }

        # Cameras
        self._process_camera_params(args)

        # Get views
        view_ids = self.data_list

        # Build dataset
        for view_id in view_ids:
            frame_id = self.get_frame_id[view_id]

            tof_filename = self._get_tof_filename(frame_id) 
            color_filename = self._get_color_filename(frame_id) 
            depth_filename = self._get_depth_filename(frame_id) 
            motion_mask_filename = self._get_motion_mask_filename(frame_id) 
            
            ## Camera params
            self.dataset['tof_intrinsics'] += [self.tof_intrinsics[view_id]]
            self.dataset['tof_extrinsics'] += [self.tof_extrinsics[view_id]]
            self.dataset['tof_poses'] += [self.tof_poses[view_id]]
            self.dataset['tof_light_poses'] += [self.tof_light_poses[view_id]]

            self.dataset['color_intrinsics'] += [self.color_intrinsics[view_id]]
            self.dataset['color_extrinsics'] += [self.tof_extrinsics[view_id]]
            self.dataset['color_poses'] += [self.color_poses[view_id]]
            self.dataset['color_light_poses'] += [self.color_light_poses[view_id]]

            ## TOF
            if args.use_tof:
                tof_im = np.squeeze(self._read_tof(tof_filename))
                tof_im = self._process_tof(tof_im)
                self.dataset['tof_images'].append(tof_im)

            ## Color
            if args.use_color:
                color_im = np.squeeze(self._read_color(color_filename))
                color_im = self._process_color(color_im)
                self.dataset['color_images'].append(color_im)

            ## Depth
            if args.use_depth or args.render_depth:
                depth_im = np.squeeze(self._read_depth(depth_filename))
                depth_im = self._process_depth(depth_im)
                self.dataset['depth_images'].append(depth_im)

            ## Motion mask
            if args.use_motion_mask:
                motion_mask = np.squeeze(self._read_motion_mask(motion_mask_filename))
                motion_mask = self._process_motion_mask(motion_mask)
                self.dataset['motion_masks'].append(motion_mask)

            ## Anything else (e.g. saving files)
            self._process_data_extra(args)

        # Post-process
        self._scale_dataset(args)
        self._stack_dataset(args)
        self._post_process_dataset(args)
        self._process_bounds(args)
    
    def _read_tof(self, tof_filename):
        return scipy.io.loadmat(tof_filename)['tof'].astype(np.float32)

    def _read_color(self, color_filename):
        return scipy.io.loadmat(color_filename)['color'].astype(np.float32)

    def _read_depth(self, depth_filename):
        return np.zeros([self.args.tof_image_height, self.args.tof_image_width], dtype=np.float32)
    
    def _read_motion_mask(self, motion_mask_filename):
        return np.ones([self.args.tof_image_height, self.args.tof_image_width], dtype=np.float32)
    
    def _process_camera_params(self, args):
        self.tof_intrinsics, self.tof_extrinsics = get_camera_params(
                os.path.join(args.source_path, f'cams/tof_intrinsics.{self.file_endings["cams"]}'),
                os.path.join(args.source_path, f'cams/tof_extrinsics.npy'),
                args
                )
        self.tof_poses = np.linalg.inv(self.tof_extrinsics)
        self.tof_poses, self.tof_tform = recenter_poses(self.tof_poses)
        self.tof_intrinsics = [np.copy(self.tof_intrinsics) for i in range(args.total_num_views)]

        self.color_intrinsics, self.color_extrinsics = get_camera_params(
                os.path.join(args.source_path, f'cams/color_intrinsics.{self.file_endings["cams"]}'),
                os.path.join(args.source_path, f'cams/color_extrinsics.npy'),
                args
                )
        self.color_poses = np.linalg.inv(self.color_extrinsics)
        self.color_poses, self.color_tform = recenter_poses(self.color_poses)
        self.color_intrinsics = [np.copy(self.color_intrinsics) for i in range(args.total_num_views)]

        tlE = get_extrinsics(
                os.path.join(args.source_path, f'cams/tof_light_extrinsics.npy'),
                args,
                default_exts=self.tof_extrinsics
                )
        self.tof_light_poses = np.linalg.inv(tlE)

        clE = get_extrinsics(
                os.path.join(args.source_path, f'cams/color_light_extrinsics.npy'),
                args,
                default_exts=self.color_extrinsics
                )
        self.color_light_poses = np.linalg.inv(clE)

        ## Depth range
        depth_range_path = os.path.join(args.source_path, f'cams/depth_range.npy')

        if os.path.exists(depth_range_path) and args.depth_range < 0:
            self.dataset['depth_range'] = np.load(depth_range_path).astype(np.float32)
        else:
            self.dataset['depth_range'] = np.array(args.depth_range).astype(np.float32)
        

    def _process_tof(self, tof_im):
        return tof_im

    def _process_color(self, color_im):
        return color_im

    def _process_depth(self, depth_im):
        return depth_im

    def _process_motion_mask(self, motion_mask):
        return motion_mask

    def _process_data_extra(self, args):
        pass
    
    def _scale_dataset(self, args):
        # Scale and crop
        if args.use_tof:
            self.dataset['tof_images'], _ = crop_mvs_input(
                    self.dataset['tof_images'],
                    self.dataset['tof_intrinsics'],
                    args.total_num_views,
                    None,
                    args.tof_image_width,
                    args.tof_image_height
                    )
            self.dataset['tof_images'], _ = scale_mvs_input(
                    self.dataset['tof_images'],
                    self.dataset['tof_intrinsics'],
                    args.total_num_views,
                    None,
                    args.tof_scale_factor
                    )
        
        if args.use_color:
            self.dataset['color_images'], _ = crop_mvs_input(
                    self.dataset['color_images'],
                    self.dataset['color_intrinsics'],
                    args.total_num_views,
                    None,
                    args.color_image_width,
                    args.color_image_height
                    )
            self.dataset['color_images'], _ = scale_mvs_input(
                    self.dataset['color_images'],
                    self.dataset['color_intrinsics'],
                    args.total_num_views,
                    None,
                    args.color_scale_factor
                    )
    
    def _stack_dataset(self, args):
        # Stack
        if args.use_tof:
            self.dataset['tof_images'] = normalize_im_max(np.stack(self.dataset['tof_images'])).astype(np.float32)
            self.dataset['tof_intrinsics'] = np.stack(self.dataset['tof_intrinsics']).astype(np.float32)
            self.dataset['tof_extrinsics'] = np.stack(self.dataset['tof_extrinsics']).astype(np.float32)
            self.dataset['tof_poses'] = np.stack(self.dataset['tof_poses']).astype(np.float32)
            self.dataset['tof_light_poses'] = np.stack(self.dataset['tof_light_poses']).astype(np.float32)

        if args.use_color:
            self.dataset['color_images'] = normalize_im_max(np.stack(self.dataset['color_images'])).astype(np.float32)
            self.dataset['color_intrinsics'] = np.stack(self.dataset['color_intrinsics']).astype(np.float32)
            self.dataset['color_extrinsics'] = np.stack(self.dataset['color_extrinsics']).astype(np.float32)
            self.dataset['color_poses'] = np.stack(self.dataset['color_poses']).astype(np.float32)
            self.dataset['color_light_poses'] = np.stack(self.dataset['color_light_poses']).astype(np.float32)
        
        if args.use_depth or args.render_depth:
            self.dataset['depth_images'] = np.stack(self.dataset['depth_images']).astype(np.float32)

        if args.use_motion_mask:
            self.dataset['motion_masks'] = np.stack(self.dataset['motion_masks']).astype(bool)

    def _post_process_dataset(self, args):
        if args.use_tof:
            self.dataset['tof_depth_images'] = np.squeeze(depth_from_tof(
                self.dataset['tof_images'], self.dataset['depth_range'], args.phase_offset
                )).astype(np.float32)
    
    def _process_bounds(self, args):
        self.dataset['bounds'] = np.stack(
            [
                args.min_depth_fac * self.dataset['depth_range'],
                args.max_depth_fac * self.dataset['depth_range']
            ],
            axis=0
            ).astype(np.float32)

    def _create_splits(self, args):
        self.dataset['i_test'] = 0

        if args.dynamic:
            self.i_train = [
                i for i in range(args.view_start, args.view_start + args.num_views * args.view_step, args.view_step)
            ]
            self.i_test = self.i_train
            self.i_val = [
                i for i in range(np.minimum(args.val_start, args.total_num_views - 1), np.minimum(args.val_end, args.total_num_views - 1))
            ]
        elif args.train_views != "":
            self.i_train = np.array([int(i) for i in args.train_views.split(",")])
            self.i_test = [i for i in np.arange(args.view_start, args.view_start + args.num_views) if (i not in self.i_train)] 
            self.i_val = self.i_test
        else:
            self.i_test = self.dataset['i_test']

            if not isinstance(self.i_test, list):
                self.i_test = [self.i_test]

            if args.autoholdout > 0:
                print('Auto holdout,', args.autoholdout)
                self.i_test = np.arange(args.num_views)[::args.autoholdout]

            self.i_val = self.i_test

            self.i_train = [
                i for i in range(args.view_start, args.view_start + args.num_views * args.view_step, args.view_step)
            ]
            self.i_train = np.array(
                [
                    i for i in self.i_train \
                        if (i not in self.i_test and i not in self.i_val) \
                        or (args.num_views == 1)
                ]
            )

        # Val frame numbers
        self.val_frames_idx = list(range(len(self.i_val)))

    def get_batch(
        self,
        i_train,
        N_rand,
        H, W,
        args,
        outputs=['tof_images', 'color_images']
        ):
        # Get image index, coordinates in image
        img_i = np.random.choice(i_train)
        coords = tf.stack(
            tf.meshgrid(
                tf.range(H),
                tf.range(W),
                indexing='ij'
                ),
            axis=-1
            )
        coords = tf.reshape(coords, [-1, 2])
        select_inds = np.random.choice(
            coords.shape[0], size=[N_rand], replace=False
            )
        select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])

        # Get batch
        batch = {}

        for k in outputs:
            if self.dataset[k][img_i] is not None:
                batch[k] = tf.gather_nd(
                    self.dataset[k][img_i],
                    select_inds
                    )
            else:
                batch[k] = None
        
        return img_i, select_inds, batch