from datetime import datetime
from utils.logger import logger, get_handler
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import ActionSenseDataset
from utils.args import init_args
from utils.utils import pformat_dict
import numpy as np
import os
import models as model_list
import tasks
import wandb
from utils.torch_device import get_device
from utils.utils import get_num_classes

# global variables among training functions
training_iterations = 0
modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)
args = None


def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used and wandb
    """
    global args
    args = init_args()
    logger.addHandler(get_handler(args.logfile))
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    # this is needed for multi-GPUs systems where you just want to use a predefined set of GPUs
    if args.gpus is not None:
        logger.debug("Using only these GPUs: {}".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    # wanbd logging configuration
    if args.wandb_name is not None:
        wandb.init(group=args.wandb_name, dir=args.wandb_dir)
        wandb.run.name = (
            args.name + "_" + args.shift.split("-")[0] + "_" + args.shift.split("-")[-1]
        )
    return args


def main():
    global training_iterations, modalities
    init_operations()
    modalities = args.modality

    # recover valid paths, domains, classes
    num_classes = get_num_classes(modalities, args.dataset.annotations_path)
    logger.info("Number of classes: {}".format(num_classes))
    # device where everything is run
    device = torch.device("cpu")
    logger.info("Device: {}".format(device))

    # these dictionaries are for more multi-modal training/testing, each key is a modality used
    models = {}
    train_transforms = {}
    test_transforms = {}
    logger.info("Instantiating models per modality")
    for m in modalities:
        logger.info("{} Net\tModality: {}".format(args.models[m].model, m))
        models[m] = getattr(model_list, args.models[m].model)(
            num_classes, m, args.models[m]
        )
        train_transforms[m], test_transforms[m] = models[m].get_augmentation(m)
    transformations = {"train": train_transforms, "test": test_transforms}
    # the models are wrapped into the ActionRecognition task which manages all the training steps
    action_classifier = tasks.ActionRecognition(
        "action-classifier",
        models,
        args.batch_size,
        args.total_batch,
        args.models_dir,
        num_classes,
        args.train.num_clips,
        args.models,
        args=args,
    )
    action_classifier.load_on_gpu(device)

    if args.action == "validate":
        if args.resume_from is not None:
            for m in modalities:
                action_classifier._Task__restore_checkpoint(m, args.resume_from[m])

        val_loader = torch.utils.data.DataLoader(
            ActionSenseDataset(
                modalities,
                "test",
                args.dataset,
                args.test,
                args.test.num_clips,
                multimodal=args.multimodal,
                transform=transformations["test"],
                load_feat=args.load_feat,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.dataset.workers,
            pin_memory=True,
            drop_last=False,
        )

        validate(
            action_classifier,
            val_loader,
            device,
            action_classifier.current_iter,
            num_classes,
        )

def validate(model, val_loader, device, it, num_classes):
    """
    function to validate the model on the test set
    model: Task containing the model to be tested
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    num_classes: int, number of classes in the classification problem
    """
    global modalities

    model.reset_acc()
    model.train(False)
    logits = {}

    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):
            label = label.to(device)

            for m in modalities:
                if m == "MIDLEVEL":
                    batch = data[m]["EMG"].shape[0]
                else:
                    batch = data[m].shape[0]
                logits[m] = torch.zeros((args.test.num_clips, batch, num_classes)).to(
                    device
                )
                if m == "EMG":
                    data[m] = data[m].reshape(
                        batch,
                        args.test.num_clips,
                        args.test.num_frames_per_clip[m],
                        -1,
                    )
                    data[m] = data[m].permute(1, 0, 2, 3)
                    assert data[m].shape == (
                        args.test.num_clips,
                        batch,
                        args.test.num_frames_per_clip[m],
                        16,
                    )
                elif m == "RGB" and args.load_feat[m]:
                    data[m] = data[m].permute(1, 0, 2)
                    assert data[m].shape == (
                        args.test.num_clips,
                        batch,
                        1024,
                    )
                elif m == "RGB" and not args.load_feat[m]:
                    raise ValueError("RGB modality should be loaded from features")
                elif m == "MIDLEVEL":
                    data[m]["EMG"] = data[m]["EMG"].reshape(
                        args.test.num_clips,
                        batch,
                        args.test.num_frames_per_clip["EMG"],
                        -1,
                    )
                    assert data[m]["EMG"].shape == (
                        args.test.num_clips,
                        batch,
                        args.test.num_frames_per_clip["EMG"],
                        16,
                    )
                    data[m]["RGB"] = data[m]["RGB"].permute(1, 0, 2)
                    assert data[m]["RGB"].shape == (
                        args.test.num_clips,
                        batch,
                        1024,
                    )

            clip = {}
            for i_c in range(args.test.num_clips):
                for m in modalities:
                    if m == "MIDLEVEL":
                        clip[m] = {}
                        for mod in ["EMG", "RGB"]:
                            clip[m][mod] = data[m][mod][i_c].to(device)
                    else:
                        clip[m] = data[m][i_c].to(device)

                output, _ = model(clip)
                for m in modalities:
                    logits[m][i_c] = output[m]

            for m in modalities:
                logits[m] = torch.mean(logits[m], dim=0)

            model.compute_accuracy(logits, label)

            if (i_val + 1) % (len(val_loader) // 5) == 0:
                logger.info(
                    "[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(
                        i_val + 1,
                        len(val_loader),
                        model.accuracy.avg[1],
                        model.accuracy.avg[5],
                    )
                )

        class_accuracies = [
            (x / y) * 100
            for x, y in zip(model.accuracy.correct, model.accuracy.total)
            if y != 0
        ]
        logger.info(
            "Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%"
            % (model.accuracy.avg[1], model.accuracy.avg[5])
        )
        for i_class, class_acc in enumerate(class_accuracies):
            logger.info(
                "Class %d = [%d/%d] = %.2f%%"
                % (
                    i_class,
                    int(model.accuracy.correct[i_class]),
                    int(model.accuracy.total[i_class]),
                    class_acc,
                )
            )

    logger.info(
        "Accuracy by averaging class accuracies (same weight for each class): {}%".format(
            np.array(class_accuracies).mean(axis=0)
        )
    )
    test_results = {
        "top1": model.accuracy.avg[1],
        "top5": model.accuracy.avg[5],
        "class_accuracies": np.array(class_accuracies),
    }

    with open(
        os.path.join(
            args.log_dir,
            f"val_precision_{str(args.modality)}-" f"{str(args.modality)}.txt",
        ),
        "a+",
    ) as f:
        f.write(
            "[%d/%d]\tAcc@top1: %.2f%% \tAcc@top5: %.2f%%\n"
            % (it, args.train.num_iter, test_results["top1"], test_results["top5"])
        )

    return test_results


if __name__ == "__main__":
    main()
