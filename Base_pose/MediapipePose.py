import cv2
import mediapipe as mp

class MediapipePose:
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.expected_keypoint_count = 33

        self.model_path = 'Base_pose/checkpoints/pose_landmarker_full.task'
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.options = self.PoseLandmarkerOptions(
        base_options=self.BaseOptions(model_asset_path=self.model_path),
        running_mode=self.VisionRunningMode.IMAGE)


    def get_features(self, image_path):
        return self._extract_pose_features_mediapipe(image_path)

    def see_features(self, image_path, pose_features):
        return self._visualize_keypoints_mediapipe(image_path, pose_features)


    def _visualize_keypoints_mediapipe(self, image_path, pose_features):
        class Landmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()

        landmarks = []
        for i in range(0, len(pose_features[0]), 3):

            landmark = Landmark(pose_features[0][i], pose_features[0][i+1], pose_features[0][i+2])
            landmarks.append(landmark)

        for landmark in landmarks:
            landmark_px = self.mp.solutions.drawing_utils._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_width, image_height)
            if landmark_px:
                cv2.circle(annotated_image, landmark_px, 20, (0, 255, 0), -1)

        return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    def _extract_pose_features_mediapipe(self, image_path):
        mp_image = mp.Image.create_from_file(image_path)

        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            results = landmarker.detect(mp_image)

        pose_landmarks = []

        if results.pose_landmarks :
                for landmark in results.pose_landmarks[0]:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    pose_landmarks.extend([x, y, z])
        return [pose_landmarks]
