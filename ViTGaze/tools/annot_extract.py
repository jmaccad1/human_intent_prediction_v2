import cv2
import json
from PIL import ImageColor
from tqdm import tqdm

class Annotator:
    """
    annotator class that extracts annotations from LabelBox ndjson files to fetch ground truths & visualizations. 
    """

    def __init__(self) -> None:
        self._annotation = None
        self._intent = None

        self._video_output = None
        self._video_capture = None

        self._save_resolution = None
        self._display_resolution = None

    def load_video(
        self,
        video_filepath: str,
        display_resolution: tuple[int],
    ) -> None:
        """Initializes cv2.VideoCapture to read video for annotation.

        Args:
            video_filepath (str): Relative filepath of video to annotation.
            display_resolution (tuple[int]): Video resolution during annotation
        """
        self._video_capture = cv2.VideoCapture(video_filepath)
        self._display_resolution = display_resolution

    def load_annotation(self, annotation_filepath: str):
        """Loads ndjson file exported from LabelBox for video annotations.
        The file contains the bounding boxes of objects detected in the video,
        along with a label of focus (isGazed) on an object with respect to the
        "head" object. This is project specific.

        Args:
            annotation_filepath (str): Relative filepath of ndjson file.
        """
        with open(annotation_filepath) as file:
            self._annotations = [json.loads(line) for line in file.readlines()]

        self._ground_truths = []
        
        # Annotation Structure for Head + Object Ground Truths
        for i in range(len(self._annotations[0]['objects'])):  # Print the first annotation to check the structure
            print(f"{i}: {self._annotations[0]['objects'][i]['title']}")  # Print the first annotation to check the structure
        
        self._intent = self._annotations[0]["classifications"][0]["answer"]
        self._intent = self._intent["value"]

    def load_output_saver(
        self, output_filepath: str, save_resolution: tuple[int]
    ) -> None:
        """Initializes cv2.VideoWriter for mp4 reading to save annotated video.

        Args:
            output_filepath (str): Relative filepath for annotated video.
            save_resolution (tuple[int]): Video resolution of annotated video.
        """
        self._video_output = cv2.VideoWriter(
            output_filepath, 0x7634706D, 30, save_resolution
        )

        self._save_resolution = save_resolution

    def _draw_bounding_box(
        self, object_name: str, bounding_box: tuple[float], color: str, frame
    ) -> None:
        """Draws a rectangular outline on a frame.

        Args:
            object_name (str):
            Label for bounding box

            bounding_box (tuple[float]): Bounding box top, left, height, width.
            Will be converted into integers for display.

            color (str): Color of bounding box in hex (#aaaaaa format)
            frame: cv2 loaded frame to annotate
        """
        color = ImageColor.getcolor(color, "RGB")[::-1]
        top, left, height, width = map(int, bounding_box.values())
        cv2.rectangle(
            frame,
            (left, top),
            (left + width, top + height),
            color,
            2,
        )

        cv2.putText(
            frame,
            object_name,
            org=(left + 7, top + 16),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=color,
        )

    def _draw_intent(self, intent: str, frame) -> None:
        """Draws text on upper left of frame.

        Args:
            intent (str): Text to annotate
            frame: cv2 loaded frame to annotate
        """
        cv2.putText(
            frame,
            intent,
            org=(30, 30),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2,
            color=(0, 0, 0),
        )

    def _draw_gaze(
        self, head_bounding_box: tuple[float], gaze_bounding_box, frame
    ) -> None:
        """Draws a line between the center point of two bounding boxes.

        Args:
            head_bounding_box (`tuple[float]`):
            Bounding box top, left, height, width of first subject.
            Will be converted into integers for display.

            gaze_bounding_box:
            Bounding box top, left, height, width of second subject.
            Will be converted into integers for display.

            frame:
            Frame loaded in cv2 to annotate
        """
        top, left, height, width = map(int, head_bounding_box.values())
        head_center = (left + (width // 2), top + (height // 2))

        top, left, height, width = map(int, gaze_bounding_box.values())
        gaze_center = (left + (width // 2), top + (height // 2))

        cv2.line(frame, head_center, gaze_center, (0, 0, 0), 2)

    def annotate(self) -> None:
        """Draws annotations on loaded video based on annotation ndjson file.

        Args:
            Display progrss bar. Defaults to False.
        """

        for annotation in tqdm(self._annotations):
            successful_read, frame = self._video_capture.read()

            self._draw_intent(self._intent, frame)

            head_bounding_box, gaze_bounding_box = None, None

            for object in annotation["objects"]:
                if object["classifications"]:
                    gaze_bounding_box = object["bbox"]

                if object["value"] == "head":
                    head_bounding_box = object["bbox"]

                self._draw_bounding_box(
                    object["value"], object["bbox"], object["color"], frame
                )

            if gaze_bounding_box is not None:
                self._draw_gaze(head_bounding_box, gaze_bounding_box, frame)

            frame = cv2.resize(frame, self._display_resolution)
            cv2.imshow("", frame)

            if self._video_output:
                frame = cv2.resize(frame, self._save_resolution)
                self._video_output.write(frame)

            escape_key = 27
            if cv2.waitKey(delay=30) & 0xFF == escape_key:
                break

    def release(self):
        """Releases video capture, video writer, and windows.
        Call after annotation or for next annotation to free resources.
        """

        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None

        if self._video_output is not None:
            self._video_output.release()
            self._video_output = None

        self._annotations = None
        self._intent = None

        cv2.destroyAllWindows()

annotation_filepath = f"./P1_V1.ndjson"
video_filepath = f"./P1_V1.mp4"
output_filepath = "./data/output.mp4"
display_resolution = "1280x720"
save_resolution = "1980x1080"
   

if __name__ == "__main__":
    display_resolution = tuple(map(int, display_resolution.split("x")))
    save_resolution = tuple(map(int, save_resolution.split("x")))

    try:   
        annotator = Annotator()
        #annotator.load_video(video_filepath, display_resolution)
        annotator.load_annotation(annotation_filepath)
        #print(annotator._intent)  # Print the intent value from the annotation file
        
        #annotator.load_output_saver(output_filepath, save_resolution)

        #annotator.annotate()
    
    finally:
        print("Done annotating video.")
        #annotator.release()