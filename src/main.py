from ImageHandler import FrameHandler
from Data.FilePaths import FILE_PATH

frame = FrameHandler(FILE_PATH, "FormCoachAI")
frame.run_video_analysis()
