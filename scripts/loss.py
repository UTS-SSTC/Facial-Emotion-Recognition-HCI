import matplotlib.pyplot as plt
import re
import numpy as np

# Set English font and style
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Training log data
training_logs = """
=== Training pipeline for microsoft/resnet-50 ===
[INFO] Loading classification and base vision models: microsoft/resnet-50 on cuda
Fine-tuning microsoft/resnet-50...
[INFO] Fine-tuning 14,981,128/23,524,424 params (63.68%)
Epoch 1/20: 100%|██████████| 700/700 [02:55<00:00,  4.00it/s]
Epoch 1: Train 2.0657, Val 2.0472, Acc 0.2172
Epoch 2/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 2: Train 1.8815, Val 1.7543, Acc 0.3231
Epoch 3/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 3: Train 1.7510, Val 1.6907, Acc 0.3497
Epoch 4/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 4: Train 1.6966, Val 1.6487, Acc 0.3696
Epoch 5/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 5: Train 1.6603, Val 1.6089, Acc 0.3871
Epoch 6/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 6: Train 1.6308, Val 1.5863, Acc 0.3997
Epoch 7/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 7: Train 1.6137, Val 1.5764, Acc 0.4005
Epoch 8/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 8: Train 1.5916, Val 1.5518, Acc 0.4108
Epoch 9/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 9: Train 1.5772, Val 1.5436, Acc 0.4169
Epoch 10/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 10: Train 1.5603, Val 1.5318, Acc 0.4218
Epoch 11/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 11: Train 1.5516, Val 1.5201, Acc 0.4259
Epoch 12/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 12: Train 1.5411, Val 1.5105, Acc 0.4288
Epoch 13/20: 100%|██████████| 700/700 [02:44<00:00,  4.26it/s]
Epoch 13: Train 1.5347, Val 1.5067, Acc 0.4304
Epoch 14/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 14: Train 1.5283, Val 1.5031, Acc 0.4325
Epoch 15/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 15: Train 1.5276, Val 1.4972, Acc 0.4349
Epoch 16/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 16: Train 1.5215, Val 1.4971, Acc 0.4327
Epoch 17/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 17: Train 1.5143, Val 1.4940, Acc 0.4369
Epoch 18/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 18: Train 1.5194, Val 1.4945, Acc 0.4363
Epoch 19/20: 100%|██████████| 700/700 [02:44<00:00,  4.24it/s]
Epoch 19: Train 1.5167, Val 1.4943, Acc 0.4363
Epoch 20/20: 100%|██████████| 700/700 [02:44<00:00,  4.25it/s]
Epoch 20: Train 1.5111, Val 1.4943, Acc 0.4363

Fine-tuning microsoft/resnet-152...
[INFO] Fine-tuning 14,981,128/58,160,200 params (25.76%)
Epoch 1/20: 100%|██████████| 700/700 [05:42<00:00,  2.05it/s]
Epoch 1: Train 2.0635, Val 2.0312, Acc 0.2347
Epoch 2/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 2: Train 1.8289, Val 1.7149, Acc 0.3349
Epoch 3/20: 100%|██████████| 700/700 [05:20<00:00,  2.19it/s]
Epoch 3: Train 1.7228, Val 1.6580, Acc 0.3648
Epoch 4/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 4: Train 1.6742, Val 1.6193, Acc 0.3821
Epoch 5/20: 100%|██████████| 700/700 [05:20<00:00,  2.19it/s]
Epoch 5: Train 1.6405, Val 1.5784, Acc 0.3995
Epoch 6/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 6: Train 1.6118, Val 1.5601, Acc 0.4088
Epoch 7/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 7: Train 1.5862, Val 1.5453, Acc 0.4176
Epoch 8/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 8: Train 1.5699, Val 1.5249, Acc 0.4233
Epoch 9/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 9: Train 1.5514, Val 1.5145, Acc 0.4282
Epoch 10/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 10: Train 1.5396, Val 1.5075, Acc 0.4250
Epoch 11/20: 100%|██████████| 700/700 [05:20<00:00,  2.19it/s]
Epoch 11: Train 1.5256, Val 1.4939, Acc 0.4381
Epoch 12/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 12: Train 1.5146, Val 1.4876, Acc 0.4380
Epoch 13/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 13: Train 1.5033, Val 1.4798, Acc 0.4435
Epoch 14/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 14: Train 1.5026, Val 1.4739, Acc 0.4462
Epoch 15/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 15: Train 1.4912, Val 1.4708, Acc 0.4479
Epoch 16/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 16: Train 1.4886, Val 1.4682, Acc 0.4495
Epoch 17/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 17: Train 1.4884, Val 1.4674, Acc 0.4499
Epoch 18/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 18: Train 1.4840, Val 1.4663, Acc 0.4511
Epoch 19/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 19: Train 1.4833, Val 1.4661, Acc 0.4502
Epoch 20/20: 100%|██████████| 700/700 [05:20<00:00,  2.18it/s]
Epoch 20: Train 1.4810, Val 1.4662, Acc 0.4502

=== Training pipeline for timm/vit_small_patch16_224.augreg_in21k ===
[INFO] Loading classification and base vision models: timm/vit_small_patch16_224.augreg_in21k on cuda
Fine-tuning timm/vit_small_patch16_224.augreg_in21k...
[INFO] Fine-tuning 3,552,008/21,668,744 params (16.39%)
Epoch 1/20: 100%|██████████| 700/700 [02:52<00:00,  4.05it/s]
Epoch 1: Train 1.8363, Val 1.6042, Acc 0.3917
Epoch 2/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 2: Train 1.5787, Val 1.5104, Acc 0.4248
Epoch 3/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 3: Train 1.5170, Val 1.4590, Acc 0.4497
Epoch 4/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 4: Train 1.4731, Val 1.4318, Acc 0.4618
Epoch 5/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 5: Train 1.4372, Val 1.3963, Acc 0.4764
Epoch 6/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 6: Train 1.4048, Val 1.3749, Acc 0.4835
Epoch 7/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 7: Train 1.3834, Val 1.3537, Acc 0.4988
Epoch 8/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 8: Train 1.3660, Val 1.3525, Acc 0.4982
Epoch 9/20: 100%|██████████| 700/700 [02:54<00:00,  4.02it/s]
Epoch 9: Train 1.3433, Val 1.3350, Acc 0.5069
Epoch 10/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 10: Train 1.3250, Val 1.3188, Acc 0.5172
Epoch 11/20: 100%|██████████| 700/700 [02:53<00:00,  4.02it/s]
Epoch 11: Train 1.3174, Val 1.3149, Acc 0.5145
Epoch 12/20: 100%|██████████| 700/700 [02:53<00:00,  4.02it/s]
Epoch 12: Train 1.3043, Val 1.3063, Acc 0.5155
Epoch 13/20: 100%|██████████| 700/700 [02:53<00:00,  4.02it/s]
Epoch 13: Train 1.2924, Val 1.3007, Acc 0.5210
Epoch 14/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 14: Train 1.2842, Val 1.2951, Acc 0.5235
Epoch 15/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 15: Train 1.2801, Val 1.2898, Acc 0.5260
Epoch 16/20: 100%|██████████| 700/700 [02:54<00:00,  4.02it/s]
Epoch 16: Train 1.2793, Val 1.2900, Acc 0.5224
Epoch 17/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 17: Train 1.2742, Val 1.2885, Acc 0.5271
Epoch 18/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 18: Train 1.2731, Val 1.2883, Acc 0.5258
Epoch 19/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 19: Train 1.2714, Val 1.2873, Acc 0.5267
Epoch 20/20: 100%|██████████| 700/700 [02:53<00:00,  4.03it/s]
Epoch 20: Train 1.2727, Val 1.2872, Acc 0.5266

Fine-tuning google/vit-base-patch16-224...
[INFO] Fine-tuning 14,181,896/85,804,808 params (16.53%)
Epoch 1/20: 100%|██████████| 700/700 [09:32<00:00,  1.22it/s]
Epoch 1: Train 1.7440, Val 1.5227, Acc 0.4173
Epoch 2/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 2: Train 1.5076, Val 1.4233, Acc 0.4591
Epoch 3/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 3: Train 1.4324, Val 1.3758, Acc 0.4795
Epoch 4/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 4: Train 1.3769, Val 1.3293, Acc 0.4995
Epoch 5/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 5: Train 1.3407, Val 1.2968, Acc 0.5142
Epoch 6/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 6: Train 1.3106, Val 1.2732, Acc 0.5204
Epoch 7/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 7: Train 1.2855, Val 1.2561, Acc 0.5284
Epoch 8/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 8: Train 1.2609, Val 1.2444, Acc 0.5332
Epoch 9/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 9: Train 1.2411, Val 1.2277, Acc 0.5393
Epoch 10/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 10: Train 1.2228, Val 1.2148, Acc 0.5466
Epoch 11/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 11: Train 1.2076, Val 1.2061, Acc 0.5499
Epoch 12/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 12: Train 1.1906, Val 1.2000, Acc 0.5486
Epoch 13/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 13: Train 1.1845, Val 1.1956, Acc 0.5530
Epoch 14/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 14: Train 1.1779, Val 1.1899, Acc 0.5539
Epoch 15/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 15: Train 1.1710, Val 1.1875, Acc 0.5581
Epoch 16/20: 100%|██████████| 700/700 [09:32<00:00,  1.22it/s]
Epoch 16: Train 1.1622, Val 1.1864, Acc 0.5581
Epoch 17/20: 100%|██████████| 700/700 [09:32<00:00,  1.22it/s]
Epoch 17: Train 1.1633, Val 1.1834, Acc 0.5596
Epoch 18/20: 100%|██████████| 700/700 [09:32<00:00,  1.22it/s]
Epoch 18: Train 1.1591, Val 1.1829, Acc 0.5596
Epoch 19/20: 100%|██████████| 700/700 [09:33<00:00,  1.22it/s]
Epoch 19: Train 1.1514, Val 1.1830, Acc 0.5599
Epoch 20/20: 100%|██████████| 700/700 [09:32<00:00,  1.22it/s]
Epoch 20: Train 1.1606, Val 1.1829, Acc 0.5602

=== Training pipeline for facebook/deit-base-distilled-patch16-224 ===
[INFO] Loading classification and base vision models: facebook/deit-base-distilled-patch16-224 on cuda
Some weights of DeiTForImageClassification were not initialized from the model checkpoint at facebook/deit-base-distilled-patch16-224 and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of DeiTModel were not initialized from the model checkpoint at facebook/deit-base-distilled-patch16-224 and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Fine-tuning facebook/deit-base-distilled-patch16-224...
[INFO] Fine-tuning 28,357,640/85,806,344 params (33.05%)
Epoch 1/20: 100%|██████████| 700/700 [11:33<00:00,  1.01it/s]
Epoch 1: Train 1.5442, Val 1.3103, Acc 0.5111
Epoch 2/20: 100%|██████████| 700/700 [11:35<00:00,  1.01it/s]
Epoch 2: Train 1.2490, Val 1.1987, Acc 0.5541
Epoch 3/20: 100%|██████████| 700/700 [11:36<00:00,  1.01it/s]
Epoch 3: Train 1.1299, Val 1.1600, Acc 0.5690
Epoch 4/20: 100%|██████████| 700/700 [11:36<00:00,  1.00it/s]
Epoch 4: Train 1.0366, Val 1.0907, Acc 0.5956
Epoch 5/20: 100%|██████████| 700/700 [11:36<00:00,  1.00it/s]
Epoch 5: Train 0.9447, Val 1.0306, Acc 0.6274
Epoch 6/20: 100%|██████████| 700/700 [11:36<00:00,  1.00it/s]
Epoch 6: Train 0.8635, Val 1.0062, Acc 0.6417
Epoch 7/20: 100%|██████████| 700/700 [11:37<00:00,  1.00it/s]
Epoch 7: Train 0.7906, Val 0.9764, Acc 0.6554
Epoch 8/20: 100%|██████████| 700/700 [11:37<00:00,  1.00it/s]
Epoch 8: Train 0.7158, Val 0.9521, Acc 0.6729
Epoch 9/20: 100%|██████████| 700/700 [11:37<00:00,  1.00it/s]
Epoch 9: Train 0.6514, Val 0.9241, Acc 0.6867
Epoch 10/20: 100%|██████████| 700/700 [11:37<00:00,  1.00it/s]
Epoch 10: Train 0.5957, Val 0.9234, Acc 0.6911
Epoch 11/20: 100%|██████████| 700/700 [11:36<00:00,  1.00it/s]
Epoch 11: Train 0.5480, Val 0.9088, Acc 0.7024
Epoch 12/20: 100%|██████████| 700/700 [11:37<00:00,  1.00it/s]
Epoch 12: Train 0.5136, Val 0.8921, Acc 0.7101
Epoch 13/20: 100%|██████████| 700/700 [11:37<00:00,  1.00it/s]
Epoch 13: Train 0.4739, Val 0.8927, Acc 0.7117
Epoch 14/20: 100%|██████████| 700/700 [11:38<00:00,  1.00it/s]
Epoch 14: Train 0.4475, Val 0.8852, Acc 0.7169
Epoch 15/20: 100%|██████████| 700/700 [11:38<00:00,  1.00it/s]
Epoch 15: Train 0.4235, Val 0.8826, Acc 0.7180
Epoch 16/20: 100%|██████████| 700/700 [11:39<00:00,  1.00it/s]
Epoch 16: Train 0.4150, Val 0.8833, Acc 0.7222
Epoch 17/20: 100%|██████████| 700/700 [11:39<00:00,  1.00it/s]
Epoch 17: Train 0.4010, Val 0.8779, Acc 0.7246
Epoch 18/20: 100%|██████████| 700/700 [11:39<00:00,  1.00it/s]
Epoch 18: Train 0.3964, Val 0.8789, Acc 0.7249
Epoch 19/20: 100%|██████████| 700/700 [11:39<00:00,  1.00it/s]
Epoch 19: Train 0.3892, Val 0.8780, Acc 0.7260
Epoch 20/20: 100%|██████████| 700/700 [11:39<00:00,  1.00it/s]
Epoch 20: Train 0.3885, Val 0.8779, Acc 0.7247


=== Training pipeline for facebook/deit-small-distilled-patch16-224 ===
[INFO] Loading classification and base vision models: facebook/deit-small-distilled-patch16-224 on cuda
Some weights of DeiTForImageClassification were not initialized from the model checkpoint at facebook/deit-small-distilled-patch16-224 and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of DeiTModel were not initialized from the model checkpoint at facebook/deit-small-distilled-patch16-224 and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Fine-tuning facebook/deit-small-distilled-patch16-224...
[INFO] Fine-tuning 10,649,864/21,669,512 params (49.15%)
Epoch 1/20: 100%|██████████| 700/700 [04:18<00:00,  2.71it/s]
Epoch 1: Train 1.6271, Val 1.3860, Acc 0.4796
Epoch 2/20: 100%|██████████| 700/700 [04:19<00:00,  2.69it/s]
Epoch 2: Train 1.3225, Val 1.2553, Acc 0.5307
Epoch 3/20: 100%|██████████| 700/700 [04:20<00:00,  2.69it/s]
Epoch 3: Train 1.2189, Val 1.2171, Acc 0.5404
Epoch 4/20: 100%|██████████| 700/700 [04:20<00:00,  2.69it/s]
Epoch 4: Train 1.1384, Val 1.1619, Acc 0.5734
Epoch 5/20: 100%|██████████| 700/700 [04:20<00:00,  2.69it/s]
Epoch 5: Train 1.0675, Val 1.1285, Acc 0.5823
Epoch 6/20: 100%|██████████| 700/700 [04:20<00:00,  2.69it/s]
Epoch 6: Train 0.9965, Val 1.0936, Acc 0.5982
Epoch 7/20: 100%|██████████| 700/700 [04:20<00:00,  2.69it/s]
Epoch 7: Train 0.9352, Val 1.0687, Acc 0.6101
Epoch 8/20: 100%|██████████| 700/700 [04:20<00:00,  2.69it/s]
Epoch 8: Train 0.8795, Val 1.0529, Acc 0.6214
Epoch 9/20: 100%|██████████| 700/700 [04:20<00:00,  2.68it/s]
Epoch 9: Train 0.8245, Val 1.0299, Acc 0.6338
Epoch 10/20: 100%|██████████| 700/700 [04:20<00:00,  2.68it/s]
Epoch 10: Train 0.7726, Val 1.0180, Acc 0.6401
Epoch 11/20: 100%|██████████| 700/700 [04:21<00:00,  2.68it/s]
Epoch 11: Train 0.7344, Val 1.0093, Acc 0.6484
Epoch 12/20: 100%|██████████| 700/700 [04:21<00:00,  2.68it/s]
Epoch 12: Train 0.6960, Val 0.9934, Acc 0.6569
Epoch 13/20: 100%|██████████| 700/700 [04:21<00:00,  2.68it/s]
Epoch 13: Train 0.6623, Val 0.9859, Acc 0.6608
Epoch 14/20: 100%|██████████| 700/700 [04:20<00:00,  2.68it/s]
Epoch 14: Train 0.6364, Val 0.9830, Acc 0.6640
Epoch 15/20: 100%|██████████| 700/700 [04:20<00:00,  2.68it/s]
Epoch 15: Train 0.6179, Val 0.9808, Acc 0.6667
Epoch 16/20: 100%|██████████| 700/700 [04:21<00:00,  2.68it/s]
Epoch 16: Train 0.5965, Val 0.9761, Acc 0.6721
Epoch 17/20: 100%|██████████| 700/700 [04:20<00:00,  2.68it/s]
Epoch 17: Train 0.5841, Val 0.9744, Acc 0.6722
Epoch 18/20: 100%|██████████| 700/700 [04:21<00:00,  2.68it/s]
Epoch 18: Train 0.5890, Val 0.9731, Acc 0.6720
Epoch 19/20: 100%|██████████| 700/700 [04:20<00:00,  2.68it/s]
Epoch 19: Train 0.5763, Val 0.9733, Acc 0.6727
Epoch 20/20: 100%|██████████| 700/700 [04:21<00:00,  2.68it/s]
Epoch 20: Train 0.5775, Val 0.9729, Acc 0.6729
Training hybrid model for facebook/deit-small-distilled-patch16-224...
[INFO] Extracting features for train set
[INFO] Extracting features for val set
[INFO] Training LightGBM classifier with 44800 samples
Training until validation scores don't improve for 20 rounds
Did not meet early stopping. Best iteration is:
[100]	train's multi_logloss: 0.574594	valid's multi_logloss: 1.04263
[INFO] Training completed in 41.52 seconds
"""

def extract_training_data(logs):
    """
    Extract training data from training logs
    Supports any number of models
    """
    pattern = r'Epoch (\d+): Train ([\d.]+), Val ([\d.]+), Acc ([\d.]+)'
    
    models_data = {}
    current_model = None
    
    lines = logs.strip().split('\n')
    
    for line in lines:
        # Detect model name
        if 'Fine-tuning ' in line and '...' in line:
            model_name = line.split('Fine-tuning ')[1].split('...')[0]
            current_model = model_name
            models_data[current_model] = {
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
                'accuracy': []
            }
        
        # Match epoch data
        match = re.search(pattern, line)
        if match and current_model:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            accuracy = float(match.group(4))
            
            models_data[current_model]['epochs'].append(epoch)
            models_data[current_model]['train_loss'].append(train_loss)
            models_data[current_model]['val_loss'].append(val_loss)
            models_data[current_model]['accuracy'].append(accuracy)
    
    return models_data

def get_model_label(model_name):
    """
    Generate simplified model labels
    """
    model_labels = {
        'microsoft/resnet-50': 'ResNet-50',
        'microsoft/resnet-152': 'ResNet-152', 
        'timm/vit_small_patch16_224.augreg_in21k': 'ViT-Small',
        'google/vit-base-patch16-224': 'ViT-Base',
        'facebook/deit-base-distilled-patch16-224': 'DeiT-Base',
        'facebook/deit-small-distilled-patch16-224': 'DeiT-Small'
    }
    return model_labels.get(model_name, model_name.split('/')[-1])

def generate_colors_and_styles(num_models):
    """
    Generate colors and line styles for any number of models
    """
    # Extended color palette
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                   '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    
    # Extended line styles
    base_line_styles = ['-', '--', '-.', ':', '-']
    
    # Generate exactly num_models colors and styles
    colors = []
    line_styles = []
    
    for i in range(num_models):
        colors.append(base_colors[i % len(base_colors)])
        line_styles.append(base_line_styles[i % len(base_line_styles)])
    
    return colors, line_styles

def plot_validation_loss(models_data, save_path='validation_loss_comparison.png'):
    """
    Plot validation loss comparison and save as separate image
    """
    plt.figure(figsize=(10, 6))
    
    num_models = len(models_data)
    colors, line_styles = generate_colors_and_styles(num_models)
    
    for i, (model_name, data) in enumerate(models_data.items()):
        label = get_model_label(model_name)
        plt.plot(data['epochs'], data['val_loss'], 
                color=colors[i], 
                linestyle=line_styles[i],
                linewidth=2.5, 
                marker='o', 
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.5,
                label=label)
    
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Validation Loss', fontweight='bold')
    plt.title('Model Training Validation Loss Comparison', fontweight='bold', pad=20)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis limits if data exists
    if models_data:
        max_epochs = max(max(data['epochs']) for data in models_data.values())
        plt.xlim(1, max_epochs)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Validation loss plot saved as: {save_path}")

def plot_accuracy(models_data, save_path='accuracy_comparison.png'):
    """
    Plot accuracy comparison and save as separate image
    """
    plt.figure(figsize=(10, 6))
    
    num_models = len(models_data)
    colors, line_styles = generate_colors_and_styles(num_models)
    
    for i, (model_name, data) in enumerate(models_data.items()):
        label = get_model_label(model_name)
        plt.plot(data['epochs'], data['accuracy'], 
                color=colors[i], 
                linestyle=line_styles[i],
                linewidth=2.5, 
                marker='s', 
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.5,
                label=label)
    
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.title('Model Training Accuracy Comparison', fontweight='bold', pad=20)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis limits if data exists
    if models_data:
        max_epochs = max(max(data['epochs']) for data in models_data.values())
        min_acc = min(min(data['accuracy']) for data in models_data.values()) - 0.02
        max_acc = min(1.0, max(max(data['accuracy']) for data in models_data.values()) + 0.02)
        plt.xlim(1, max_epochs)
        plt.ylim(min_acc, max_acc)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Accuracy plot saved as: {save_path}")

def print_summary(models_data):
    """
    Print training results summary
    """
    print("\n=== Training Results Summary ===")
    print(f"{'Model Name':<30} {'Final Val Loss':<15} {'Final Accuracy':<15} {'Best Accuracy':<15}")
    print("-" * 75)
    
    # Sort models by final accuracy (descending)
    sorted_models = sorted(models_data.items(), 
                          key=lambda x: x[1]['accuracy'][-1], 
                          reverse=True)
    
    for model_name, data in sorted_models:
        final_loss = data['val_loss'][-1]
        final_acc = data['accuracy'][-1]
        best_acc = max(data['accuracy'])
        label = get_model_label(model_name)
        print(f"{label:<30} {final_loss:<15.4f} {final_acc:<15.4f} {best_acc:<15.4f}")
    
    print(f"\nTotal models analyzed: {len(models_data)}")

def main():
    """
    Main function to process training logs and generate plots
    """
    print("Extracting training data from logs...")
    models_data = extract_training_data(training_logs)
    
    if not models_data:
        print("No training data found in logs!")
        return
    
    print(f"Found {len(models_data)} models:")
    for model_name in models_data.keys():
        print(f"  - {get_model_label(model_name)}")
    
    print("\nGenerating validation loss comparison plot...")
    plot_validation_loss(models_data)
    
    print("Generating accuracy comparison plot...")
    plot_accuracy(models_data)
    
    # Print summary
    print_summary(models_data)

if __name__ == "__main__":
    main()