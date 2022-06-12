import CycleGAN as cg

content_img_path = '../dataset/tempA/*.png'
style_img_path = '../dataset/tempB/*.png'
output_folder = './results/'
checkpoint_filepath = output_folder+"{epoch}_epoch/model_checkpoints/cyclegan_checkpoints.{epoch}"

weight_file_path = cg.train(content_img_path, style_img_path, checkpoint_filepath, output_folder, 2)

input_img_path = '../dataset/test_A/inputs/*.png'
# weight_file_path = output_folder+"20_epoch/model_checkpoints/cyclegan_checkpoints.20"

cg.test(input_img_path, weight_file_path, output_folder)