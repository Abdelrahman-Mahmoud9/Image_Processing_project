import src.camera_calibrator as ccb
import src.final_pipeline as fnp
img_file_pattern='./camera_cal/calibration*.jpg'
objpoints, imgpoints, img_size = ccb.find_chessboard_corners(img_file_pattern, nx=9, ny=6)
mtx, dist = ccb.calibrate_camera(img_file_pattern,nx=9,  ny=6)
input_path=input("enter the path of the input file : ")
output_path=input("enter the path of the output file : ")
fnp.pipeline_for_video(mtx, dist, input_video=input_path,output_video=output_path)