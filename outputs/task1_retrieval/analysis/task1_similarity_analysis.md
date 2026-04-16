# Task 1 相似度矩阵与案例分析

## lavis_blip2_feature_extractor_pretrain_n1000_mean_mean_7c8f5056

- 模型：`lavis:blip2_feature_extractor:pretrain`
- Text-to-Image：R@1=69.62, R@5=92.06, R@10=96.84
- Image-to-Text：R@1=79.80, R@5=95.70, R@10=98.60
- 相似度矩阵图：[lavis_blip2_feature_extractor_pretrain_n1000_mean_mean_7c8f5056_sim_matrix.png](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/lavis_blip2_feature_extractor_pretrain_n1000_mean_mean_7c8f5056_sim_matrix.png)

### Text-to-Image 成功样例

- 查询：`A big burly grizzly bear is show with grass in the background.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.4178`
- 查询：`The large brown bear has a black nose.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.3550`
- 查询：`Closeup of a brown bear sitting in a grassy area.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.4069`
- 查询：`A large bear that is sitting on grass.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.3892`
- 查询：`A close up picture of a brown bear's face.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.3896`

### Text-to-Image 失败样例

- 查询：`A woman stands in the dining area at the table.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000048396.jpg`
  - Correct：`False`
  - Score：`0.3536`
- 查询：`A room with chairs, a table, and a woman in it.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000107339.jpg`
  - Correct：`False`
  - Score：`0.3690`
- 查询：`A woman standing in a kitchen by a window`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000045229.jpg`
  - Correct：`False`
  - Score：`0.3876`
- 查询：`A person standing at a table in a room.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000112298.jpg`
  - Correct：`False`
  - Score：`0.3708`
- 查询：`A living area with a television and a table`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000097337.jpg`
  - Correct：`False`
  - Score：`0.3460`

### Image-to-Text 成功样例

- 查询：`000000000285.jpg`
  - Ground truth：`A big burly grizzly bear is show with grass in the background.`
  - Prediction：`A big burly grizzly bear is show with grass in the background.`
  - Correct：`True`
  - Score：`0.4178`
- 查询：`000000000632.jpg`
  - Ground truth：`Bedroom scene with a bookcase, blue comforter and window.`
  - Prediction：`Bedroom scene with a bookcase, blue comforter and window.`
  - Correct：`True`
  - Score：`0.4147`
- 查询：`000000000724.jpg`
  - Ground truth：`A stop sign is mounted upside-down on it's post.`
  - Prediction：`An upside down stop sign by the road.`
  - Correct：`True`
  - Score：`0.4112`
- 查询：`000000000776.jpg`
  - Ground truth：`Three teddy bears, each a different color, snuggling together.`
  - Prediction：`Three teddy bears, each a different color, snuggling together.`
  - Correct：`True`
  - Score：`0.4078`
- 查询：`000000000802.jpg`
  - Ground truth：`A kitchen with a refrigerator, stove and oven with cabinets.`
  - Prediction：`Stark white appliances stand out against brown wooden cabinets.`
  - Correct：`True`
  - Score：`0.4485`

### Image-to-Text 失败样例

- 查询：`000000000139.jpg`
  - Ground truth：`A woman stands in the dining area at the table.`
  - Prediction：`A brightly decorated living room with a stylish feel.`
  - Correct：`False`
  - Score：`0.4323`
- 查询：`000000000785.jpg`
  - Ground truth：`A woman posing for the camera standing on skis.`
  - Prediction：`The woman is learning how to ski on the snow.`
  - Correct：`False`
  - Score：`0.4010`
- 查询：`000000000885.jpg`
  - Ground truth：`a male tennis player in white shorts is playing tennis`
  - Prediction：`A picture of a person playing in a tennis game.`
  - Correct：`False`
  - Score：`0.4327`
- 查询：`000000001503.jpg`
  - Ground truth：`A computer on a desk next to a laptop.`
  - Prediction：`A laptop computer and a desktop computer on a white desk`
  - Correct：`False`
  - Score：`0.4195`
- 查询：`000000002153.jpg`
  - Ground truth：`Batter preparing to swing at pitch during major game.`
  - Prediction：`The ump and baseball players on the field.`
  - Correct：`False`
  - Score：`0.4268`

## lavis_blip2_feature_extractor_pretrain_n1000_first_first_74077bce

- 模型：`lavis:blip2_feature_extractor:pretrain`
- Text-to-Image：R@1=48.81, R@5=75.13, R@10=84.25
- Image-to-Text：R@1=63.70, R@5=88.80, R@10=94.80
- 相似度矩阵图：[lavis_blip2_feature_extractor_pretrain_n1000_first_first_74077bce_sim_matrix.png](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/lavis_blip2_feature_extractor_pretrain_n1000_first_first_74077bce_sim_matrix.png)

### Text-to-Image 成功样例

- 查询：`A living area with a television and a table`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000000139.jpg`
  - Correct：`True`
  - Score：`0.4236`
- 查询：`Bedroom scene with a bookcase, blue comforter and window.`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.3391`
- 查询：`A bedroom with a bookshelf full of books.`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.3228`
- 查询：`This room has a bed with blue sheets and a large bookcase`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.3138`
- 查询：`a bed room with a neatly made bed a window and a book shelf`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.3753`

### Text-to-Image 失败样例

- 查询：`A woman stands in the dining area at the table.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000048396.jpg`
  - Correct：`False`
  - Score：`0.3119`
- 查询：`A room with chairs, a table, and a woman in it.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000107339.jpg`
  - Correct：`False`
  - Score：`0.3476`
- 查询：`A woman standing in a kitchen by a window`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000045229.jpg`
  - Correct：`False`
  - Score：`0.3606`
- 查询：`A person standing at a table in a room.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000107339.jpg`
  - Correct：`False`
  - Score：`0.3121`
- 查询：`A big burly grizzly bear is show with grass in the background.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000050638.jpg`
  - Correct：`False`
  - Score：`0.2285`

### Image-to-Text 成功样例

- 查询：`000000000632.jpg`
  - Ground truth：`Bedroom scene with a bookcase, blue comforter and window.`
  - Prediction：`a bed room with a neatly made bed a window and a book shelf`
  - Correct：`True`
  - Score：`0.3753`
- 查询：`000000000802.jpg`
  - Ground truth：`A kitchen with a refrigerator, stove and oven with cabinets.`
  - Prediction：`A white oven and a white refrigerator are in the kitchen.`
  - Correct：`True`
  - Score：`0.4612`
- 查询：`000000001296.jpg`
  - Ground truth：`A woman holding a Hello Kitty phone on her hands.`
  - Prediction：`A woman in white shirt holding up a cellphone.`
  - Correct：`True`
  - Score：`0.2351`
- 查询：`000000001353.jpg`
  - Ground truth：`some children are riding on a mini orange train`
  - Prediction：`Several children on a small  indoor kiddie train.`
  - Correct：`True`
  - Score：`0.3059`
- 查询：`000000001584.jpg`
  - Ground truth：`The red, double decker bus is driving past other buses.`
  - Prediction：`The red, double decker bus is driving past other buses.`
  - Correct：`True`
  - Score：`0.4281`

### Image-to-Text 失败样例

- 查询：`000000000139.jpg`
  - Ground truth：`A woman stands in the dining area at the table.`
  - Prediction：`a room with a fire place and television inside of it`
  - Correct：`False`
  - Score：`0.4260`
- 查询：`000000000285.jpg`
  - Ground truth：`A big burly grizzly bear is show with grass in the background.`
  - Prediction：`I am unable to see the image above.`
  - Correct：`False`
  - Score：`0.1791`
- 查询：`000000000724.jpg`
  - Ground truth：`A stop sign is mounted upside-down on it's post.`
  - Prediction：`There is a street sign on top of a stop sign near a tree.`
  - Correct：`False`
  - Score：`0.3897`
- 查询：`000000000776.jpg`
  - Ground truth：`Three teddy bears, each a different color, snuggling together.`
  - Prediction：`A white, beige and brown baby bear under a beige/white comforter.`
  - Correct：`False`
  - Score：`0.3037`
- 查询：`000000000785.jpg`
  - Ground truth：`A woman posing for the camera standing on skis.`
  - Prediction：`a woman is wearing a pink and black jacket is sking`
  - Correct：`False`
  - Score：`0.2628`

## lavis_blip2_feature_extractor_pretrain_n1000_mean_first_ee4d0a29

- 模型：`lavis:blip2_feature_extractor:pretrain`
- Text-to-Image：R@1=71.86, R@5=92.90, R@10=97.10
- Image-to-Text：R@1=76.10, R@5=95.30, R@10=98.30
- 相似度矩阵图：[lavis_blip2_feature_extractor_pretrain_n1000_mean_first_ee4d0a29_sim_matrix.png](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/lavis_blip2_feature_extractor_pretrain_n1000_mean_first_ee4d0a29_sim_matrix.png)

### Text-to-Image 成功样例

- 查询：`A big burly grizzly bear is show with grass in the background.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.4226`
- 查询：`The large brown bear has a black nose.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.3750`
- 查询：`Closeup of a brown bear sitting in a grassy area.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.4021`
- 查询：`A large bear that is sitting on grass.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.3755`
- 查询：`A close up picture of a brown bear's face.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.3627`

### Text-to-Image 失败样例

- 查询：`A woman stands in the dining area at the table.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000048396.jpg`
  - Correct：`False`
  - Score：`0.3415`
- 查询：`A room with chairs, a table, and a woman in it.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000107339.jpg`
  - Correct：`False`
  - Score：`0.3849`
- 查询：`A woman standing in a kitchen by a window`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000045229.jpg`
  - Correct：`False`
  - Score：`0.3971`
- 查询：`A person standing at a table in a room.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000112298.jpg`
  - Correct：`False`
  - Score：`0.3712`
- 查询：`A living area with a television and a table`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000013923.jpg`
  - Correct：`False`
  - Score：`0.4118`

### Image-to-Text 成功样例

- 查询：`000000000285.jpg`
  - Ground truth：`A big burly grizzly bear is show with grass in the background.`
  - Prediction：`A big burly grizzly bear is show with grass in the background.`
  - Correct：`True`
  - Score：`0.4226`
- 查询：`000000000632.jpg`
  - Ground truth：`Bedroom scene with a bookcase, blue comforter and window.`
  - Prediction：`a bed room with a neatly made bed a window and a book shelf`
  - Correct：`True`
  - Score：`0.4420`
- 查询：`000000000776.jpg`
  - Ground truth：`Three teddy bears, each a different color, snuggling together.`
  - Prediction：`Three teddy bears, each a different color, snuggling together.`
  - Correct：`True`
  - Score：`0.4223`
- 查询：`000000000802.jpg`
  - Ground truth：`A kitchen with a refrigerator, stove and oven with cabinets.`
  - Prediction：`Stark white appliances stand out against brown wooden cabinets.`
  - Correct：`True`
  - Score：`0.4791`
- 查询：`000000000872.jpg`
  - Ground truth：`A couple of baseball player standing on a field.`
  - Prediction：`Two guys playing baseball, with trees in the back.`
  - Correct：`True`
  - Score：`0.4170`

### Image-to-Text 失败样例

- 查询：`000000000139.jpg`
  - Ground truth：`A woman stands in the dining area at the table.`
  - Prediction：`A brightly decorated living room with a stylish feel.`
  - Correct：`False`
  - Score：`0.4655`
- 查询：`000000000724.jpg`
  - Ground truth：`A stop sign is mounted upside-down on it's post.`
  - Prediction：`A stop sign on a post at a public street.`
  - Correct：`False`
  - Score：`0.4366`
- 查询：`000000000785.jpg`
  - Ground truth：`A woman posing for the camera standing on skis.`
  - Prediction：`A woman on skis on a ski slope posing for a picture.`
  - Correct：`False`
  - Score：`0.4091`
- 查询：`000000000885.jpg`
  - Ground truth：`a male tennis player in white shorts is playing tennis`
  - Prediction：`A picture of a person playing in a tennis game.`
  - Correct：`False`
  - Score：`0.4249`
- 查询：`000000002153.jpg`
  - Ground truth：`Batter preparing to swing at pitch during major game.`
  - Prediction：`The ump and baseball players on the field.`
  - Correct：`False`
  - Score：`0.4461`

## lavis_blip2_feature_extractor_pretrain_nall_first_first_91ffb98f

- 模型：`lavis:blip2_feature_extractor:pretrain`
- Text-to-Image：R@1=30.03, R@5=52.09, R@10=62.17
- Image-to-Text：R@1=40.38, R@5=68.06, R@10=78.60
- 相似度矩阵图：[lavis_blip2_feature_extractor_pretrain_nall_first_first_91ffb98f_sim_matrix.png](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/lavis_blip2_feature_extractor_pretrain_nall_first_first_91ffb98f_sim_matrix.png)

### Text-to-Image 成功样例

- 查询：`A living area with a television and a table`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000000139.jpg`
  - Correct：`True`
  - Score：`0.4236`
- 查询：`Bedroom scene with a bookcase, blue comforter and window.`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.3392`
- 查询：`A bedroom with a bookshelf full of books.`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.3229`
- 查询：`a bed room with a neatly made bed a window and a book shelf`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.3753`
- 查询：`A kitchen with a refrigerator, stove and oven with cabinets.`
  - Ground truth：`000000000802.jpg`
  - Prediction：`000000000802.jpg`
  - Correct：`True`
  - Score：`0.4190`

### Text-to-Image 失败样例

- 查询：`A woman stands in the dining area at the table.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000564336.jpg`
  - Correct：`False`
  - Score：`0.3365`
- 查询：`A room with chairs, a table, and a woman in it.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000492758.jpg`
  - Correct：`False`
  - Score：`0.4110`
- 查询：`A woman standing in a kitchen by a window`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000369503.jpg`
  - Correct：`False`
  - Score：`0.3698`
- 查询：`A person standing at a table in a room.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000371699.jpg`
  - Correct：`False`
  - Score：`0.3662`
- 查询：`A big burly grizzly bear is show with grass in the background.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000513484.jpg`
  - Correct：`False`
  - Score：`0.3623`

### Image-to-Text 成功样例

- 查询：`000000000632.jpg`
  - Ground truth：`Bedroom scene with a bookcase, blue comforter and window.`
  - Prediction：`a bed room with a neatly made bed a window and a book shelf`
  - Correct：`True`
  - Score：`0.3753`
- 查询：`000000000802.jpg`
  - Ground truth：`A kitchen with a refrigerator, stove and oven with cabinets.`
  - Prediction：`A white oven and a white refrigerator are in the kitchen.`
  - Correct：`True`
  - Score：`0.4612`
- 查询：`000000001353.jpg`
  - Ground truth：`some children are riding on a mini orange train`
  - Prediction：`Several children on a small  indoor kiddie train.`
  - Correct：`True`
  - Score：`0.3057`
- 查询：`000000001584.jpg`
  - Ground truth：`The red, double decker bus is driving past other buses.`
  - Prediction：`The red, double decker bus is driving past other buses.`
  - Correct：`True`
  - Score：`0.4281`
- 查询：`000000001761.jpg`
  - Ground truth：`Two planes flying in the sky over a bridge.`
  - Prediction：`Two planes fly over a bridge in Sydney, Australia, with the Sydney Opera House in the background.`
  - Correct：`True`
  - Score：`0.4420`

### Image-to-Text 失败样例

- 查询：`000000000139.jpg`
  - Ground truth：`A woman stands in the dining area at the table.`
  - Prediction：`A TV sitting in a living room on a hard wood floor.`
  - Correct：`False`
  - Score：`0.4487`
- 查询：`000000000285.jpg`
  - Ground truth：`A big burly grizzly bear is show with grass in the background.`
  - Prediction：`The brown fuzzy dog has round blue eyes.`
  - Correct：`False`
  - Score：`0.1925`
- 查询：`000000000724.jpg`
  - Ground truth：`A stop sign is mounted upside-down on it's post.`
  - Prediction：`There is a truck behind the stop sign.`
  - Correct：`False`
  - Score：`0.4197`
- 查询：`000000000776.jpg`
  - Ground truth：`Three teddy bears, each a different color, snuggling together.`
  - Prediction：`A white, beige and brown baby bear under a beige/white comforter.`
  - Correct：`False`
  - Score：`0.3036`
- 查询：`000000000785.jpg`
  - Ground truth：`A woman posing for the camera standing on skis.`
  - Prediction：`The woman using the skies is wearing all black.`
  - Correct：`False`
  - Score：`0.3342`

## lavis_blip_feature_extractor_base_nall_first_first_69e93c04

- 模型：`lavis:blip_feature_extractor:base`
- Text-to-Image：R@1=49.90, R@5=76.11, R@10=84.88
- Image-to-Text：R@1=63.76, R@5=86.80, R@10=93.12
- 相似度矩阵图：[lavis_blip_feature_extractor_base_nall_first_first_69e93c04_sim_matrix.png](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/lavis_blip_feature_extractor_base_nall_first_first_69e93c04_sim_matrix.png)

### Text-to-Image 成功样例

- 查询：`A big burly grizzly bear is show with grass in the background.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.3318`
- 查询：`Closeup of a brown bear sitting in a grassy area.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.3466`
- 查询：`A large bear that is sitting on grass.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.3381`
- 查询：`A close up picture of a brown bear's face.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.3142`
- 查询：`Bedroom scene with a bookcase, blue comforter and window.`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.3193`

### Text-to-Image 失败样例

- 查询：`A woman stands in the dining area at the table.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000564336.jpg`
  - Correct：`False`
  - Score：`0.2619`
- 查询：`A room with chairs, a table, and a woman in it.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000441247.jpg`
  - Correct：`False`
  - Score：`0.2728`
- 查询：`A woman standing in a kitchen by a window`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000455597.jpg`
  - Correct：`False`
  - Score：`0.2844`
- 查询：`A person standing at a table in a room.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000371699.jpg`
  - Correct：`False`
  - Score：`0.2745`
- 查询：`A living area with a television and a table`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000368900.jpg`
  - Correct：`False`
  - Score：`0.2814`

### Image-to-Text 成功样例

- 查询：`000000000285.jpg`
  - Ground truth：`A big burly grizzly bear is show with grass in the background.`
  - Prediction：`Closeup of a brown bear sitting in a grassy area.`
  - Correct：`True`
  - Score：`0.3466`
- 查询：`000000000632.jpg`
  - Ground truth：`Bedroom scene with a bookcase, blue comforter and window.`
  - Prediction：`a bed room with a neatly made bed a window and a book shelf`
  - Correct：`True`
  - Score：`0.3261`
- 查询：`000000000776.jpg`
  - Ground truth：`Three teddy bears, each a different color, snuggling together.`
  - Prediction：`Three teddy bears, each a different color, snuggling together.`
  - Correct：`True`
  - Score：`0.2954`
- 查询：`000000000872.jpg`
  - Ground truth：`A couple of baseball player standing on a field.`
  - Prediction：`Two guys playing baseball, with trees in the back.`
  - Correct：`True`
  - Score：`0.3148`
- 查询：`000000001000.jpg`
  - Ground truth：`The people are posing for a group photo.`
  - Prediction：`A group of kids posing for a picture on a tennis court.`
  - Correct：`True`
  - Score：`0.3390`

### Image-to-Text 失败样例

- 查询：`000000000139.jpg`
  - Ground truth：`A woman stands in the dining area at the table.`
  - Prediction：`The spaceous living room has a large television and a fireplace.`
  - Correct：`False`
  - Score：`0.2900`
- 查询：`000000000724.jpg`
  - Ground truth：`A stop sign is mounted upside-down on it's post.`
  - Prediction：`a stop sign a building cars bushes and trees`
  - Correct：`False`
  - Score：`0.3039`
- 查询：`000000000785.jpg`
  - Ground truth：`A woman posing for the camera standing on skis.`
  - Prediction：`A woman is on skis on the ski slope.`
  - Correct：`False`
  - Score：`0.3199`
- 查询：`000000000802.jpg`
  - Ground truth：`A kitchen with a refrigerator, stove and oven with cabinets.`
  - Prediction：`A kitchen has a plain white fridge in the corner.`
  - Correct：`False`
  - Score：`0.3198`
- 查询：`000000000885.jpg`
  - Ground truth：`a male tennis player in white shorts is playing tennis`
  - Prediction：`A tennis player makes a quick shuffle to return the ball.`
  - Correct：`False`
  - Score：`0.3245`

## lavis_blip_retrieval_coco_nall_first_first_2abfcb45

- 模型：`lavis:blip_retrieval:coco`
- Text-to-Image：R@1=64.96, R@5=86.74, R@10=92.39
- Image-to-Text：R@1=83.20, R@5=96.36, R@10=98.20
- 相似度矩阵图：[lavis_blip_retrieval_coco_nall_first_first_2abfcb45_sim_matrix.png](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/lavis_blip_retrieval_coco_nall_first_first_2abfcb45_sim_matrix.png)

### Text-to-Image 成功样例

- 查询：`A big burly grizzly bear is show with grass in the background.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.4454`
- 查询：`Closeup of a brown bear sitting in a grassy area.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.4536`
- 查询：`A large bear that is sitting on grass.`
  - Ground truth：`000000000285.jpg`
  - Prediction：`000000000285.jpg`
  - Correct：`True`
  - Score：`0.4302`
- 查询：`Bedroom scene with a bookcase, blue comforter and window.`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.4794`
- 查询：`A bedroom with a bookshelf full of books.`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.4313`

### Text-to-Image 失败样例

- 查询：`A woman stands in the dining area at the table.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000441247.jpg`
  - Correct：`False`
  - Score：`0.3841`
- 查询：`A room with chairs, a table, and a woman in it.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000492758.jpg`
  - Correct：`False`
  - Score：`0.4389`
- 查询：`A woman standing in a kitchen by a window`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000369503.jpg`
  - Correct：`False`
  - Score：`0.3811`
- 查询：`A person standing at a table in a room.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000485844.jpg`
  - Correct：`False`
  - Score：`0.3683`
- 查询：`A living area with a television and a table`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000029596.jpg`
  - Correct：`False`
  - Score：`0.4057`

### Image-to-Text 成功样例

- 查询：`000000000139.jpg`
  - Ground truth：`A woman stands in the dining area at the table.`
  - Prediction：`A room with chairs, a table, and a woman in it.`
  - Correct：`True`
  - Score：`0.4057`
- 查询：`000000000285.jpg`
  - Ground truth：`A big burly grizzly bear is show with grass in the background.`
  - Prediction：`Closeup of a brown bear sitting in a grassy area.`
  - Correct：`True`
  - Score：`0.4536`
- 查询：`000000000632.jpg`
  - Ground truth：`Bedroom scene with a bookcase, blue comforter and window.`
  - Prediction：`Bedroom scene with a bookcase, blue comforter and window.`
  - Correct：`True`
  - Score：`0.4794`
- 查询：`000000000724.jpg`
  - Ground truth：`A stop sign is mounted upside-down on it's post.`
  - Prediction：`An upside down stop sign by the road.`
  - Correct：`True`
  - Score：`0.4290`
- 查询：`000000000776.jpg`
  - Ground truth：`Three teddy bears, each a different color, snuggling together.`
  - Prediction：`Three teddy bears, each a different color, snuggling together.`
  - Correct：`True`
  - Score：`0.4717`

### Image-to-Text 失败样例

- 查询：`000000000802.jpg`
  - Ground truth：`A kitchen with a refrigerator, stove and oven with cabinets.`
  - Prediction：`A kitchen area features a white refrigerator a stove and other appliances and brown cabinets.`
  - Correct：`False`
  - Score：`0.4439`
- 查询：`000000001503.jpg`
  - Ground truth：`A computer on a desk next to a laptop.`
  - Prediction：`A laptop computer and a desktop computer on a white desk`
  - Correct：`False`
  - Score：`0.4373`
- 查询：`000000002153.jpg`
  - Ground truth：`Batter preparing to swing at pitch during major game.`
  - Prediction：`A pitcher throwing a baseball toward a batter.`
  - Correct：`False`
  - Score：`0.4155`
- 查询：`000000002299.jpg`
  - Ground truth：`Many small children are posing together in the black and white photo.`
  - Prediction：`A group of children pose for a class picture.`
  - Correct：`False`
  - Score：`0.4733`
- 查询：`000000002431.jpg`
  - Ground truth：`A plate on a wooden table full of bread.`
  - Prediction：`A plate of cheese bread next to bread sticks and wine.`
  - Correct：`False`
  - Score：`0.4224`

## lavis_clip_feature_extractor_ViT-B-32_nall_first_first_c1e57b48

- 模型：`lavis:clip_feature_extractor:ViT-B-32`
- Text-to-Image：R@1=30.36, R@5=54.78, R@10=66.09
- Image-to-Text：R@1=50.02, R@5=75.00, R@10=83.26
- 相似度矩阵图：[lavis_clip_feature_extractor_ViT-B-32_nall_first_first_c1e57b48_sim_matrix.png](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/lavis_clip_feature_extractor_ViT-B-32_nall_first_first_c1e57b48_sim_matrix.png)

### Text-to-Image 成功样例

- 查询：`Bedroom scene with a bookcase, blue comforter and window.`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.3548`
- 查询：`A bedroom with a bookshelf full of books.`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.3379`
- 查询：`This room has a bed with blue sheets and a large bookcase`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.3341`
- 查询：`a bed room with a neatly made bed a window and a book shelf`
  - Ground truth：`000000000632.jpg`
  - Prediction：`000000000632.jpg`
  - Correct：`True`
  - Score：`0.3406`
- 查询：`A group of people that are standing near a tennis net.`
  - Ground truth：`000000001000.jpg`
  - Prediction：`000000001000.jpg`
  - Correct：`True`
  - Score：`0.3066`

### Text-to-Image 失败样例

- 查询：`A woman stands in the dining area at the table.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000189475.jpg`
  - Correct：`False`
  - Score：`0.2959`
- 查询：`A room with chairs, a table, and a woman in it.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000228144.jpg`
  - Correct：`False`
  - Score：`0.2934`
- 查询：`A woman standing in a kitchen by a window`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000491216.jpg`
  - Correct：`False`
  - Score：`0.3151`
- 查询：`A person standing at a table in a room.`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000244750.jpg`
  - Correct：`False`
  - Score：`0.2984`
- 查询：`A living area with a television and a table`
  - Ground truth：`000000000139.jpg`
  - Prediction：`000000097337.jpg`
  - Correct：`False`
  - Score：`0.3240`

### Image-to-Text 成功样例

- 查询：`000000000285.jpg`
  - Ground truth：`A big burly grizzly bear is show with grass in the background.`
  - Prediction：`The large brown bear has a black nose.`
  - Correct：`True`
  - Score：`0.3259`
- 查询：`000000000632.jpg`
  - Ground truth：`Bedroom scene with a bookcase, blue comforter and window.`
  - Prediction：`Bedroom scene with a bookcase, blue comforter and window.`
  - Correct：`True`
  - Score：`0.3548`
- 查询：`000000000776.jpg`
  - Ground truth：`Three teddy bears, each a different color, snuggling together.`
  - Prediction：`A group of three stuffed animal teddy bears.`
  - Correct：`True`
  - Score：`0.3241`
- 查询：`000000000785.jpg`
  - Ground truth：`A woman posing for the camera standing on skis.`
  - Prediction：`A woman in a red jacket skiing down a slope`
  - Correct：`True`
  - Score：`0.3453`
- 查询：`000000001000.jpg`
  - Ground truth：`The people are posing for a group photo.`
  - Prediction：`A group of people that are standing near a tennis net.`
  - Correct：`True`
  - Score：`0.3066`

### Image-to-Text 失败样例

- 查询：`000000000139.jpg`
  - Ground truth：`A woman stands in the dining area at the table.`
  - Prediction：`A large living room is seen in this image.`
  - Correct：`False`
  - Score：`0.3027`
- 查询：`000000000724.jpg`
  - Ground truth：`A stop sign is mounted upside-down on it's post.`
  - Prediction：`A stop sign with the words "Don't" and "Believing" added.`
  - Correct：`False`
  - Score：`0.3577`
- 查询：`000000000802.jpg`
  - Ground truth：`A kitchen with a refrigerator, stove and oven with cabinets.`
  - Prediction：`An older picture of a large kitchen with white appliances.`
  - Correct：`False`
  - Score：`0.3453`
- 查询：`000000000872.jpg`
  - Ground truth：`A couple of baseball player standing on a field.`
  - Prediction：`A player running the bases of a base ball game while an opposing player goes for the ball.`
  - Correct：`False`
  - Score：`0.3251`
- 查询：`000000000885.jpg`
  - Ground truth：`a male tennis player in white shorts is playing tennis`
  - Prediction：`A player winds up to serve a tennis ball to his opponent.`
  - Correct：`False`
  - Score：`0.3460`
