import cv2
import pyvista as pv
import tkinter as tk
from tkinter import filedialog
from cvzone.FaceDetectionModule import FaceDetector

def load_new_model(plotter):
    file_path = filedialog.askopenfilename(
        title="Seleccionar modelo CAD",
        filetypes=[("Archivos CAD", "*.stl *.obj *.ply"), ("Todos los archivos", "*.*")]
    )
    if file_path:
        plotter.clear()
        mesh = pv.read(file_path)
        plotter.add_mesh(mesh, color='lightblue', specular=0.5, specular_power=15)
        plotter.reset_camera()
        plotter.update()

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Seleccionar modelo CAD",
    filetypes=[("Archivos CAD", "*.stl *.obj *.ply"), ("Todos los archivos", "*.*")]
)

if not file_path:
    print("No se seleccionó archivo. Saliendo...")
    exit()

plotter = pv.Plotter()
mesh = pv.read(file_path)
plotter.add_mesh(mesh, color='lightblue', specular=0.5, specular_power=15)

plotter.camera_position = 'xy'
plotter.camera.azimuth = 0
plotter.camera.elevation = 0
plotter.camera.roll = 0
plotter.show(interactive_update=True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


detector = FaceDetector(minDetectionCon=0.8)

SENSITIVITY = 0.01

# Bucle principal
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Voltear el frame horizontalmente
    img, bboxs = detector.findFaces(frame)

    if bboxs:
        center = bboxs[0]['center']
        x, y = center[0], center[1]

        # Normalizar las coordenadas x e y al rango [-1, 1]
        x_norm = (x / 640) * 2 - 1
        y_norm = (y / 480) * 2 - 1

        plotter.camera.position = (
            x_norm * 100,
            -y_norm * 100,
            plotter.camera.position[2]  # Mantener la profundidad fija
        )
        plotter.update()

    cv2.imshow('Seguimiento de Cabeza', img)

    key = cv2.waitKey(1)
    if key == ord('e'):  # Salir
        break
    elif key == ord('c'):  # Cargar nuevo modelo
        load_new_model(plotter)
    elif key == ord('r'):  # Resetear cámara
        plotter.reset_camera()
        plotter.update()

cap.release()
cv2.destroyAllWindows()
plotter.close()