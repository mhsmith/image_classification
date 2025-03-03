import cv2
import toga
from toga.style import Pack

from .image_classifier import ImageClassifier, ImageClassifierOptions


class ImageClassification(toga.App):
    def startup(self):
        self.classifier = ImageClassifier(
            str(self.paths.app / "efficientnet_lite0.tflite"),
            options=ImageClassifierOptions(max_results=4)
        )

        main_box = toga.Box(
            style=Pack(direction="column", alignment="center", padding=10),
            children=[
                toga.ImageView(id="image", style=Pack(flex=1)),
                toga.Label(
                    "Press below to take a photo.",
                    id="result",
                    style=Pack(text_align="center", font_size=16),
                ),
                toga.Box(style=Pack(flex=1)),  # Spacer
                toga.Button("Take photo", on_press=self.take_photo),
            ],
        )

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

        self.process_photo(toga.Image(self.paths.app / "fox-landscape.jpg"))

    async def take_photo(self, widget):
        try:
            if not self.camera.has_permission:
                await self.camera.request_permission()
            image = await self.camera.take_photo()
        except PermissionError as e:
            await self.main_window.dialog(
                toga.InfoDialog("PermissionError", str(e))
            )
            return

        if image:
            self.process_photo(image)

    def process_photo(self, image):
        self.widgets["image"].image = image

        self.paths.data.mkdir(exist_ok=True)
        path = str(self.paths.data / "tmp.jpg")
        image.save(path)

        opencv_image = cv2.imread(path)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        categories = self.classifier.classify(opencv_image)

        self.widgets["result"].text = "\n" + "\n".join(
            f"{cat.label}: {round(cat.score * 100)}%"
            for cat in categories
        )


def main():
    return ImageClassification()
