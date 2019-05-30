#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os.path as osp
import time
# import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from data.data import grantXdatasetGene
from data.dataForClassify import grantXdatasetForClassify
from model.fusenet import FuseNet
from data.data import writeShape
import shapefile
from pyproj import Proj, transform
import random
import gdal
import time
import os
import json
import pandas
from os.path import join


def load_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    network.load_state_dict(torch.load(save_path))


def save_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    torch.save(network.to("cpu").state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()


def poly_lr_scheduler(optimizer, init_lr, epoch, max_epoch, power):
    new_lr = init_lr * (1 - float(epoch) / max_epoch) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def classifyAllImg(dataset, batchsize, model, device, criterion, branch):
    dataloaderClassify = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        num_workers=20,
        shuffle=False,
    )

    model.eval()
    startTime = time.time()
    with torch.no_grad():
        noneVal = 0
        name = ['Water', 'Bog',
                'Channel_Fen', 'Forest_Dense',
                'Forest_Sparse', 'Wetland']
        colors = np.array([(0, 0, 255), (255, 255, 255), (255, 255, 0),
                           (46, 139, 87), (60, 179, 113), (255, 0, 0), (0, 0, 0)], dtype=np.uint8)
        rows, cols = dataset.RC
        lossMap = np.zeros([rows, cols, 6], dtype=np.float32)
        rgbMap = np.zeros([3, rows, cols], dtype=np.uint8)

        batches = len(dataloaderClassify)
        for i, sample in enumerate(dataloaderClassify):

            images1 = sample['image1'].to(device)
            images2 = sample['image2'].to(device)
            coordinates = sample['coordinate'].tolist()
            imageFlags = sample['imageFlag'].tolist()

            # if there are invalid tensors
            invalidFlag = not min(imageFlags)

            # if not in rgb image
            if invalidFlag:
                newImages1 = []
                newImages2 = []
                newCoordinates = []
                # process the invalid frame out of the batch
                for image1, image2, coordinate, flag in zip(images1, images2, coordinates, imageFlags):
                    if flag:
                        newImages1.append(image1)
                        newImages2.append(image2)
                        newCoordinates.append(coordinate)
                    if not flag:
                        cntCol, cntRow = coordinate
                        # invalid coordinate to be color[6]
                        rgbMap[:, cntRow, cntCol] = colors[6]
                images1 = torch.stack(newImages1)
                images2 = torch.stack(newImages2)
                coordinates = newCoordinates

            # if batch still has element
            # outputs = inferWithAug(model, images)
            if branch == 'resnet1':
                images = images1
            elif branch == 'resnet2':
                images = images2
            else:
                assert False

            outputs = getattr(model, branch)(images)
            _, predictions = torch.max(outputs.data, 1)

            # use fake label to calculate loss for each output
            losses = []
            for cntImg in range(outputs.shape[0]):
                losses.append(float(criterion(outputs[cntImg:cntImg + 1, ...], predictions[cntImg:cntImg + 1, ...])))

            predictions = predictions.to("cpu").tolist()

            for loss, coordinate, prediction, predictionVec in zip(losses, coordinates, predictions,
                                                                   outputs.data.to("cpu").tolist()):
                cntCol, cntRow = coordinate
                lossMap[cntRow, cntCol, :] = predictionVec
                rgbMap[:, cntRow, cntCol] = colors[int(prediction)]
                costTime = time.time() - startTime
                print("process batch ({}/{}), cost sec {}, predict remain sec {}".format
                      (i + 1, batches, int(costTime), int(costTime / (i + 1) * (batches - (i + 1)))))

    # write the tiff image
    originX, originY = dataset.XY
    pixelWidth, pixelHeight = dataset.pWH

    projectionSrc = "PROJCS[\"WGS 84 / UTM zone 10N\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",-123],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"32610\"]]"
    dst_filename = 'testOut/classifyResLateFuse.tiff'

    x_pixels = cols  # number of pixels in x
    y_pixels = rows  # number of pixels in y

    driver = gdal.GetDriverByName('GTiff')

    geotif = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        3,
        gdal.GDT_Byte, )

    geotif.SetGeoTransform((
        originX,  # 0
        pixelWidth,  # 1
        0,  # 2
        originY,  # 3
        0,  # 4
        pixelHeight))

    geotif.SetProjection(projectionSrc)
    geotif.GetRasterBand(1).WriteArray(rgbMap[0, ...].astype(np.uint8))
    geotif.GetRasterBand(2).WriteArray(rgbMap[1, ...].astype(np.uint8))
    geotif.GetRasterBand(3).WriteArray(rgbMap[2, ...].astype(np.uint8))
    geotif.FlushCache()  # Write to disk.

    # write loss map
    pdHDF = pandas.HDFStore("lossMapFake.hdf")
    lossMap = lossMap.reshape(lossMap.shape[0] * lossMap.shape[1], lossMap.shape[2])
    pdHDF.append("lossMapFake", pandas.DataFrame(lossMap))
    pdHDF.close()

    return


def inferWithAug(model, images1, images2, mode='whole'):
    """
    :param model:
    :param images1:
    :param images2:
    :param mode:
    :return: outputs
    """
    if mode == 'whole':
        prediction1 = model(images1, images2)
        prediction2 = model(torch.flip(images1, [2]), torch.flip(images2, [2]))
        prediction3 = model(torch.flip(images1, [3]), torch.flip(images2, [3]))
        prediction4 = model(torch.flip(images1, [2, 3]), torch.flip(images2, [2, 3]))
    elif mode == 'ensemble':
        prediction1 = model.resnet1(images1) + model.resnet2(images2)
        prediction2 = model.resnet1(torch.flip(images1, [2])) + model.resnet2(torch.flip(images2, [2]))
        prediction3 = model.resnet1(torch.flip(images1, [3])) + model.resnet2(torch.flip(images2, [3]))
        prediction4 = model.resnet1(torch.flip(images1, [2, 3])) + model.resnet2(torch.flip(images2, [2, 3]))
    elif mode == 'resnet1':
        prediction1 = model.resnet1(images1)
        prediction2 = model.resnet1(torch.flip(images1, [2]))
        prediction3 = model.resnet1(torch.flip(images1, [3]))
        prediction4 = model.resnet1(torch.flip(images1, [2, 3]))
    else:
        prediction1 = model.resnet2(images2)
        prediction2 = model.resnet2(torch.flip(images2, [2]))
        prediction3 = model.resnet2(torch.flip(images2, [3]))
        prediction4 = model.resnet2(torch.flip(images2, [2, 3]))

    outputs = (prediction1 + prediction2 + prediction3 + prediction4)/4
    return outputs


def test(device, dataloader, model, validation=True, criterion=None, mode='whole'):
    num_class = 6
    name = ['Water', 'Bog',
            'Channel_Fen', 'Forest_Dense',
            'Forest_Sparse', 'Wetland']

    if validation:
        print("#### Validation ####")
    else:
        print("####### Test #######")

    model.eval()

    correct = np.zeros(num_class, dtype=int)
    total = np.zeros(num_class, dtype=int)

    shapeCorrect = []
    shapeWrong = []
    loss_batch = []
    with torch.no_grad():
        for data in dataloader:
            images1, images2, labels = data['image1'], data['image2'], data['label']
            images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)
            shapes = data['originShape'].tolist()
            outputs = inferWithAug(model, images1, images2, mode=mode)
            if criterion is not None:
                for i in range(images1.shape[0]):
                    loss_batch.append(float(criterion(outputs[i:i+1, ...], labels[i:i+1, ...])))
            _, predicted = torch.max(outputs.data, 1)

            labels_arr = labels.tolist()
            preds_arr = predicted.tolist()
            for i in range(len(labels_arr)):
                label = labels_arr[i]
                total[label] += 1
                correct[label] += 1 if labels_arr[i] == preds_arr[i] else 0
                if labels_arr[i] == preds_arr[i]:
                    shapeCorrect.append(shapes[i])
                else:
                    shapeWrong.append(shapes[i])

    # write shape
    # writeShape(shapeCorrect, 'correctShapes')
    # writeShape(shapeWrong, 'wrongShapes')

    width = 10
    name_width = 24

    print()
    print(' ' + '-' * (5 + name_width + width))
    print('| {0: >{name_width}} | {1: >{width}} |'.format(
        ' ', 'Accuracy', name_width=name_width, width=width))
    print(' ' + '-' * (5 + name_width + width))
    for i in range(num_class):
        acc = 0 if total[i] == 0 else (correct[i] / total[i])
        print('| {0: >{name_width}} | {1: >{width}.6f} |'.format(
            name[i], acc, name_width=name_width, width=width))
    print(' ' + '-' * (5 + name_width + width))
    print('Accuracy of {}: {:.6f}'.format(
        'validation' if validation else 'testing',
        np.sum(correct) / np.sum(total)))

    model.train()

    # if criterion is not None:
    #     print("loss of batch:")
    #     for loss in loss_batch:
    #         print(loss)
    #     print(np.mean(np.array(loss_batch)))
    #     print(np.max(np.array(loss_batch)))
    #     print(np.mean(np.array(loss_batch)))

    return {'accuracy': np.sum(correct) / np.sum(total), 'accuracyVec': correct/total}


def main():
    config = "config/config_singleBranch.yaml"
    cuda = True
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    # Configuration
    CONFIG = Dict(yaml.load(open(config)))
    CONFIG.SAVE_DIR = osp.join('checkpoints', CONFIG.EXPERIENT)
    CONFIG.LOGNAME = join(CONFIG.SAVE_DIR, 'log.txt')

    if CONFIG.BRANCH == "rgb":
        BRANCHNAME = "resnet1"
    elif CONFIG.BRANCH == "ndn":
        BRANCHNAME = "resnet2"
    else:
        raise Exception("BRANCH should be one of rgb and ndn.")

    # Dataset
    datasetTrainValis, datasetTest = grantXdatasetGene(CONFIG.ROOT, CONFIG.CROPSIZE)

    # DataLoader
    dataloaderTrain = []
    dataloaderVali = []
    for i in range(3):
        dataloaderTrain.append(
            torch.utils.data.DataLoader(
                dataset=datasetTrainValis['train'+str(i)],
                batch_size=CONFIG.BATCH_SIZE,
                num_workers=CONFIG.NUM_WORKERS,
                shuffle=True,
            )
        )

        dataloaderVali.append(
            torch.utils.data.DataLoader(
                dataset=datasetTrainValis['vali'+str(i)],
                batch_size=CONFIG.BATCH_SIZE,
                num_workers=CONFIG.NUM_WORKERS,
                shuffle=False,
            )
        )

    dataloaderTest = torch.utils.data.DataLoader(
                dataset=datasetTest,
                batch_size=CONFIG.BATCH_SIZE,
                num_workers=CONFIG.NUM_WORKERS,
                shuffle=False,
            )

    # Model
    model = FuseNet("lateFuse", num_classes=CONFIG.N_CLASSES)
    model.to(device)
    # for name, param in model.named_parameters():
    #     if param.requires_grad and ('layer1' in name or 'layer2' in name in name):
    #         param.requires_grad = False
    #         print(name)

    # Loss definition
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    if CONFIG.TESTON:
        accuracyList = []
        accuracyVecList = []
        for cntDataset in range(3):
            load_network(join(CONFIG.SAVE_DIR, str(cntDataset)), getattr(model, BRANCHNAME),
                         "xgrant_classification_res50_resnet", "highAcc")
            result = test(device, dataloaderTest, model, mode=BRANCHNAME)
            print("accuracy is {}, accuracy vector is {}".format(result['accuracy'], result['accuracyVec']))
            accuracyList.append(result['accuracy'])
            accuracyVecList.append(result['accuracyVec'])
        print("average accuracy is {}, average accuracy vector is {}".format(np.mean(accuracyVecList),
                                                                             np.mean(accuracyVecList, axis=0)))
        return

    if CONFIG.MAPGENEON:
        datasetClassify = grantXdatasetForClassify(CONFIG.ROOT, CONFIG.CROPSIZE,
                                                   datasetTest.imageRGBfilenames,
                                                   datasetTest.rgbContours,
                                                   datasetTest.rgbXYWHCR,
                                                   datasetTest.depthImgFilename,
                                                   datasetTest.depthXYWHCR)

        classifyAllImg(datasetClassify, CONFIG.BATCH_SIZE, model, device, criterion, BRANCHNAME)
        return

    # 3 datasets for cross validation train
    # first, detect four model saving folder
    if os.path.exists(CONFIG.SAVE_DIR):
        print("the folder of this experiment already exists, Please check if it's safe to overwrite")
        return
    else:
        print("making directory for result saving")
        os.makedirs(CONFIG.SAVE_DIR)
        for i in range(3):
            os.makedirs(join(CONFIG.SAVE_DIR, str(i)))

    accuracyVec = []
    eachAccuracyVec = []
    for cntDataset in range(3):
        max_test_accuracy = 0
        print("train on datasets{}".format(cntDataset),
              file=open(CONFIG.LOGNAME, "a"))
        # model
        model = FuseNet("lateFuse", num_classes=CONFIG.N_CLASSES)
        model.to(device)

        # Optimizer
        optimizer = torch.optim.SGD(
                # cf lr_mult and decay_mult in train.prototxt
                params=[
                    {
                        "params": model.parameters(),
                        "lr": CONFIG.LR,
                        "weight_decay": CONFIG.WEIGHT_DECAY,
                    }
                ],
                momentum=CONFIG.MOMENTUM,
            )

        #visualizer
        # vis = Visualizer(CONFIG.DISPLAYPORT)
        getattr(model, BRANCHNAME).train()
        iter_start_time = time.time()
        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()
        for epoch in range(CONFIG.EPOCH_START, CONFIG.EPOCH_MAX + 1):
            # Set a learning rate
            poly_lr_scheduler(
                optimizer=optimizer,
                init_lr=CONFIG.LR,
                epoch=epoch - 1,
                max_epoch=CONFIG.EPOCH_MAX,
                power=CONFIG.POLY_POWER,
            )
            for iteration, sample in enumerate(dataloaderTrain[cntDataset]):

                iter_loss = 0

                # Image
                image1, image2, label = sample['image1'], sample['image2'], sample['label']

                if CONFIG.BRANCH == "rgb":
                    image = image1
                elif CONFIG.BRANCH == "ndn":
                    image = image2
                else:
                    raise Exception("BRANCH should be one of rgb and ndn.")

                image = image.to(device)
                label = label.to(device)

                # Propagate forward
                output = getattr(model, BRANCHNAME)(image)
                # Loss
                loss = criterion(output, label)
                # Backpropagate (just compute gradients wrt the loss)
                loss.backward()

                iter_loss += float(loss)

                # Update weights with accumulated gradients
                optimizer.step()
                # Visualizer and Summery Writer
                if iteration % CONFIG.ITER_TF == 0:
                    print("epoch {}, itr {}, loss is {}".format(epoch, iteration, iter_loss))
                    print("epoch {}, itr {}, loss is {}".format(epoch, iteration, iter_loss),
                          file=open(CONFIG.LOGNAME, "a"))  #
                    # print("time taken for each iter is %.3f" % ((time.time() - iter_start_time)/iteration))
                    # vis.drawLine(torch.FloatTensor([iteration]), torch.FloatTensor([iter_loss]))
                    # vis.displayImg(inputImgTransBack(data), classToRGB(outputs[3][0].to("cpu").max(0)[1]),
                    #                classToRGB(target[0].to("cpu")))
            # test a model
            if epoch % 10 == 0:
                result = test(device, dataloaderVali[cntDataset], model, mode=BRANCHNAME)
                getattr(model, BRANCHNAME).train()
                print("accuracy is {} in epoch {}".format(result['accuracy'], epoch),
                      file=open(CONFIG.LOGNAME, "a"))
                if result['accuracy'] > max_test_accuracy:
                    max_test_accuracy = result['accuracy']
                    max_accVec = result['accuracyVec']
                    save_network(join(CONFIG.SAVE_DIR, str(cntDataset)),
                                 getattr(model, BRANCHNAME), "xgrant_classification_res50_resnet", "highAcc")

        print("highest accuracy is {}".format(max_test_accuracy))
        print("highest accuracy is {}".format(max_test_accuracy),
              file=open(CONFIG.LOGNAME, "a"))
        accuracyVec.append(max_test_accuracy)
        eachAccuracyVec.append(max_accVec)
        print("The average accuracy on all datasets is {}, average accuracy for each class is {}".
              format(np.mean(accuracyVec), np.mean(eachAccuracyVec, axis=0)),
              file=open(CONFIG.LOGNAME, "a"))


if __name__ == "__main__":
    main()
