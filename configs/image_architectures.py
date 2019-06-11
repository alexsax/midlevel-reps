# Habitat configs
#   This should be sourced by the training script,
#   which must save a sacred experiment in the variable "ex"
#   For descriptions of all fields, see configs/core.py


@ex.named_config
def cfg_scratch():
    # Do no preprocessing to the input image (besides resizing and rescaling)

    image_dim = 84
    cfg = {}
    cfg['env'] = {
        'collate_env_obs': False,
        'transform_fn': """
            TransformFactory.independent(
            {{
                'rgb_filled':rescale_centercrop_resize((3, {image_dim}, {image_dim})),
                'target':identity_transform()
            }},
            keep_unnamed=False)""".format(image_dim=image_dim),

    }
    cfg['learner'] = { 'perception_network': 'scratch' }

    
@ex.named_config
def cfg_taskonomy_encoding():
    # Transforms RGB images to the intermediate representation from one of the networks from the paper:
    #    Taskonomy: Disentangling Task Transfer Learning (Zamir et al. '18)

    cfg = {}
    cfg['learner'] = { 'perception_network': 'features_only',
                       'taskonomy_encoder': '/mnt/models/normal_encoder.dat',
                       'encoder_type': 'taskonomy'
                       }
    cfg['env'] = {
        'collate_env_obs': True,
        'transform_fn': """
            TransformFactory.independent(
            {{
                'taskonomy':taskonomy_features_transform_collated('{taskonomy_encoder}',encoder_type='{encoder_type}'),
                'target':identity_transform()
            }},
            keep_unnamed=False)
            """.translate(remove_whitespace).format(
                encoder_type=cfg['learner']['encoder_type'],
                taskonomy_encoder=cfg['learner']['taskonomy_encoder']),
    }



@ex.named_config
def cfg_taskonomy_decoding():
    # Transforms RGB images to the output from one of the networks from the paper:
    #    Taskonomy: Disentangling Task Transfer Learning (Zamir et al. '18)

    cfg = {}
    cfg['env'] = {
        'collate_env_obs': False,
        'transform_fn': """
            TransformFactory.independent(
            {{
                'rgb_filled':cross_modal_transform(TaskonomyNetwork(load_encoder_path='/mnt/models/normal_encoder.dat',
                                                            load_decoder_path='/mnt/models/normal_decoder.dat').cuda().eval(),
                                                            (3,{image_dim}, {image_dim})),
                'target':identity_transform()
            }},
            keep_unnamed=False)
            """.format(
                encoder_type=cfg['learner']['encoder_type'],
                taskonomy_encoder=cfg['learner']['taskonomy_encoder'],
                image_dim=84),
        }
    cfg['learner'] = { 'perception_network': 'scratch' }


@ex.named_config
def cfg_taskonomy_decoding_collate():
    # Transforms RGB images to the output from one of the networks from the paper:
    #    Taskonomy: Disentangling Task Transfer Learning (Zamir et al. '18)
    # Collated versions are slightly faster, at the cost of using slightly more GPU memory

    image_dim = 84
    cfg = {}
    cfg['learner'] = { 'perception_network': 'scratch',
                       'taskonomy_encoder': '/mnt/models/normal_encoder.dat',
                       'encoder_type': 'taskonomy'
                       }
    cfg['env'] = {
        'collate_env_obs': True,
        'transform_fn': """
            TransformFactory.independent(
            {{
                'rgb_filled':cross_modal_transform_collated(TaskonomyNetwork(
                    load_encoder_path='/mnt/models/normal_encoder.dat',
                    load_decoder_path='/mnt/models/normal_decoder.dat').cuda().eval(),
                    (3,{image_dim},{image_dim})),
                'target':identity_transform()
            }},
            keep_unnamed=False)""".format(
                encoder_type=cfg['learner']['encoder_type'],
                taskonomy_encoder=cfg['learner']['taskonomy_encoder'],
                image_dim=image_dim),
    }

    
@ex.named_config
def cfg_unet_decoding():
    # Transforms RGB images to the output from a trained UNet

    image_dim = 84
    cfg = {}
    cfg['env'] = {
        'collate_env_obs': False,
        'transform_fn': """
            TransformFactory.independent(
            {{
                'rgb_filled':cross_modal_transform(load_from_file(
                        UNet(),
                        '/mnt/logdir/homoscedastic_normal_regression-checkpoints-ckpt-4.dat').cuda().eval(),
                        (3,{image_dim}, {image_dim})),
                'target':identity_transform()
            }},
            keep_unnamed=False)""".format(
                encoder_type=cfg['learner']['encoder_type'],
                taskonomy_encoder=cfg['learner']['taskonomy_encoder'],
                image_dim=image_dim), 
    }
    cfg['learner'] = { 'perception_network': 'scratch' }


