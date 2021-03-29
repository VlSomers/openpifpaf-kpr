"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os
import copy

import PIL
import math
import torch
import numpy as np
import matplotlib
import cv2

from . import datasets, decoder, network, show, transforms, visualizer, __version__

LOG = logging.getLogger(__name__)

try:
    import matplotlib.cm
    CMAP_JET = copy.copy(matplotlib.cm.get_cmap('jet'))
    CMAP_JET.set_bad('white', alpha=0.5)
except ImportError:
    CMAP_JET = None

def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.predict',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    network.cli(parser)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.1, seed_threshold=0.5)
    show.cli(parser)
    visualizer.cli(parser)
    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('--show', default=False, action='store_true',
                        help='show image of output overlay')
    parser.add_argument('--image-output', default=None, nargs='?', const=True,
                        help='image output file or directory')
    parser.add_argument('--fields-output', default=None, nargs='?', const=True,
                        help='pifpaf fields output file or directory')
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='json output file or directory')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='processing batch size')
    parser.add_argument('--long-edge', default=None, type=int,
                        help='apply preprocessing to batch images')
    parser.add_argument('--loader-workers', default=None, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--line-width', default=6, type=int,
                        help='line width for skeleton')
    parser.add_argument('--monocolor-connections', default=False, action='store_true')
    parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
    parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
    group = parser.add_argument_group('logging')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    group.add_argument('--debug-images', default=False, action='store_true',
                       help='print debug messages and enable all debug images')
    args = parser.parse_args()

    if args.debug_images:
        args.debug = True

    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig()
    logging.getLogger('openpifpaf').setLevel(log_level)
    LOG.setLevel(log_level)

    network.configure(args)
    show.configure(args)
    visualizer.configure(args)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    # glob
    if args.glob:
        # print("GLOOOOOOOOOOOB")
        # print(args.glob)
        print(glob.glob(args.glob))
        args.images += glob.glob(args.glob)
    if not args.images:
        # print(args.glob)
        raise Exception("no image files given")

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    return args


def processor_factory(args):
    # load model
    model_cpu, _ = network.factory_from_args(args)
    model = model_cpu.to(args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.base_net = model_cpu.base_net
        model.head_nets = model_cpu.head_nets
    processor = decoder.factory_from_args(args, model)
    return processor, model


def preprocess_factory(args):
    preprocess = [transforms.NormalizeAnnotations()]
    if args.long_edge:
        preprocess.append(transforms.RescaleAbsolute(args.long_edge))
    if args.batch_size > 1:
        assert args.long_edge, '--long-edge must be provided for batch size > 1'
        preprocess.append(transforms.CenterPad(args.long_edge))
    else:
        preprocess.append(transforms.CenterPadTight(16))
    return transforms.Compose(preprocess + [transforms.EVAL_TRANSFORM])


def out_name(arg, in_name, default_extension):
    """Determine an output name from args, input name and extension.

    arg can be:
    - none: return none (e.g. show image but don't store it)
    - True: activate this output and determine a default name
    - string:
        - not a directory: use this as the output file name
        - is a directory: use directory name and input name to form an output
    """
    if arg is None:
        return None

    if arg is True:
        return in_name + default_extension

    if os.path.isdir(arg):
        return os.path.join(
            arg,
            os.path.basename(in_name)
        ) + default_extension

    return arg


def main():
    args = cli()

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList(args.images, preprocess=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_anns_meta)

    # visualizers
    keypoint_painter = show.KeypointPainter(
        color_connections=not args.monocolor_connections,
        linewidth=args.line_width,
    )
    annotation_painter = show.AnnotationPainter(keypoint_painter=keypoint_painter)

    for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
        pred_batch, fields_batch = processor.batch(model, image_tensors_batch, device=args.device)

        # unbatch
        for pred, meta, fields in zip(pred_batch, meta_batch, fields_batch):
            LOG.info('batch %d: %s', batch_i, meta['file_name'])

            # load the original image if necessary
            cpu_image = None
            if args.debug or args.show or args.image_output is not None:
                with open(meta['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

            visualizer.BaseVisualizer.image(cpu_image)
            if preprocess is not None:
                pred = preprocess.annotations_inverse(pred, meta)

            if args.json_output is not None:
                json_out_name = out_name(
                    args.json_output, meta['file_name'], '.predictions.json')
                LOG.debug('json output = %s', json_out_name)
                with open(json_out_name, 'w') as f:
                    json.dump([ann.json_data() for ann in pred], f)

            if args.show or args.image_output is not None:
                image_out_name = out_name(
                    args.image_output, meta['file_name'], '.predictions.jpg')
                LOG.debug('image output = %s', image_out_name)
                with show.image_canvas(cpu_image,
                                       image_out_name,
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    annotation_painter.annotations(ax, pred)

            if args.fields_output is not None:

                paf_confidence_map_out_name = out_name(args.fields_output, meta['file_name'], '.paf_max.jpg')
                LOG.debug('PAF confidence map output = %s', paf_confidence_map_out_name)
                paf_confidence_map = np.max(fields[1][:, 0, :, :], 0)
                # paf_confidence_map = fields[1][1, 0, :, :]
                show_heatmap(cpu_image, paf_confidence_map, paf_confidence_map_out_name)

                pif_confidence_map_out_name = out_name(args.fields_output, meta['file_name'], '.pif_max.jpg')
                LOG.debug('PIF confidence map output = %s', pif_confidence_map_out_name)
                pif_confidence_map = np.max(fields[0][:, 0, :, :], 0)
                show_heatmap(cpu_image, pif_confidence_map, pif_confidence_map_out_name)


                # paf_confidence_map_out_name = out_name(args.fields_output, meta['file_name'], '.paf_sum.jpg')
                # LOG.debug('PAF confidence map output = %s', paf_confidence_map_out_name)
                # paf_confidence_map = np.sum(fields[1][:, 0, :, :], 0)
                # # paf_confidence_map = fields[1][1, 0, :, :]
                # show_heatmap(cpu_image, paf_confidence_map, paf_confidence_map_out_name)
                #
                # pif_confidence_map_out_name = out_name(args.fields_output, meta['file_name'], '.pif_sum.jpg')
                # LOG.debug('PIF confidence map output = %s', pif_confidence_map_out_name)
                # pif_confidence_map = np.sum(fields[0][:, 0, :, :], 0)
                # show_heatmap(cpu_image, pif_confidence_map, pif_confidence_map_out_name)

                # fields_out_name = out_name(True, meta['file_name'], '.fields.npy')
                # LOG.debug('fields output = %s', fields_out_name)
                # np.save(fields_out_name, np.asanyarray(fields))

                ## Dump pif map :
                pif_map = fields[0]
                pif_map_out_name = out_name(args.fields_output, meta['file_name'], '.pif.npy')
                LOG.debug('pif map output = %s', pif_map_out_name)
                np.save(pif_map_out_name, pif_map)

                ## Dump paf map :
                paf_map = fields[1]
                paf_map_out_name = out_name(args.fields_output, meta['file_name'], '.paf.npy')
                LOG.debug('paf map output = %s', paf_map_out_name)
                np.save(paf_map_out_name, paf_map)


                # ## Dump confidence fields :
                # confidence_fields = np.concatenate((fields[0][:, 0, :, :], fields[1][:, 0, :, :]), axis=0)
                # confidence_fields_out_name = out_name(True, meta['file_name'], '.confidence_fields.npy')
                # LOG.debug('confidence fields output = %s', confidence_fields_out_name)
                # np.save(confidence_fields_out_name, confidence_fields)

                # DISPLAY BODY PARTS HEATMAPS INDIVIDUALLY
                # for i in range(0, fields[0].shape[0]):
                #     confidence_map_out_name = out_name(True, meta['file_name'], '.pif_confidence_field_' + str(i) + '.jpg')
                #     LOG.debug('confidence fields output = %s', confidence_map_out_name)
                #     show_heatmap(cpu_image, fields[0][i, 0, :, :], confidence_map_out_name)
                #
                # for i in range(0, fields[1].shape[0]):
                #     confidence_map_out_name = out_name(True, meta['file_name'], '.paf_confidence_field_' + str(i) + '.png')
                #     LOG.debug('confidence fields output = %s', confidence_map_out_name)
                #     show_heatmap(cpu_image, fields[1][i, 0, :, :], confidence_map_out_name)




                # confidence_fields_compressed_out_name = out_name(True, meta['file_name'], '.confidence_fields_compressed.npy')
                # np.savez_compressed(confidence_fields_compressed_out_name, confidence_fields)
                #
                #
                # test_out_name = out_name(True, meta['file_name'], '.image.npy')
                # np.save(test_out_name, np.asanyarray(cpu_image))
                # compress_out_name = out_name(True, meta['file_name'], '.image_compressed.npy')
                # np.savez_compressed(compress_out_name, np.asanyarray(cpu_image))

                # fields[1, :, 0]
                # im = ax.imshow(self.scale_scalar(confidences[f], self.stride),
                #                alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_BLUES_NAN)
                # self.colorbar(ax, im)

                ################################
                # CODE FOR GENERATING BOUNDING BOXES AND POSE FROM WIDE IMAGES
                # for i, ann in enumerate(pred):
                #     # get bounding box
                #     x, y, w, h = ann.bbox()
                #     xs = x
                #     ys = y
                #     xe = x+w
                #     ye = y+h
                #     scale = 0.1
                #     xs = max(0, xs - w*scale)
                #     ys = max(0, ys - h*scale)
                #     xe = min(cpu_image.size[0], xe + w*scale)
                #     ye = min(cpu_image.size[1], ye + h*scale)
                #     xs = math.floor(xs)
                #     ys = math.floor(ys)
                #     xe = math.ceil(xe)
                #     ye = math.ceil(ye)
                #
                #     # crop fields and image
                #     image_crop = cpu_image.crop((xs, ys, xe, ye))
                #     confidence_fields = np.concatenate((fields[0][:, 0, :, :], fields[1][:, 0, :, :]), axis=0)
                #     confidence_fields = np.transpose(confidence_fields, (1, 2, 0))
                #     confidence_fields = cv2.resize(confidence_fields, dsize=cpu_image.size, interpolation=cv2.INTER_LINEAR)
                #     confidence_fields = np.transpose(confidence_fields, (2, 0, 1))
                #     confidence_fields_crop = confidence_fields[:, ys:ye, xs:xe]
                #
                #     # write crop image to disk
                #     path, filename = os.path.split(meta['file_name'])
                #     dirname = os.path.join(os.path.dirname(path), "bounding_boxes")
                #     if not os.path.exists(dirname):
                #         os.makedirs(dirname)
                #     image_crop_out_name = os.path.join(dirname, '{}_bb_{}.png'.format(os.path.splitext(filename)[0], i))
                #     LOG.debug('image_crop output = %s', image_crop_out_name)
                #     image_crop.save(image_crop_out_name)
                #
                #     # write crop fields to disk
                #     fields_crop_out_name = image_crop_out_name + '.confidence_fields.npy'
                #     LOG.debug('fields_crop output = %s', fields_crop_out_name)
                #     np.save(fields_crop_out_name, confidence_fields_crop)


def show_heatmap(cpu_image, paf_confidence_map, paf_confidence_map_out_name):
    with show.image_canvas(cpu_image, fig_file=paf_confidence_map_out_name, margin=[0.0, 0.01, 0.05, 0.01], show=False) as ax:
        ax.imshow(visualizer.base.BaseVisualizer.scale_scalar(paf_confidence_map, 8), alpha=0.4, vmin=0.0, vmax=1.0,
                  cmap=CMAP_JET)


if __name__ == '__main__':
    main()
