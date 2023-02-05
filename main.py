from threading import Thread, Lock
import matplotlib.pyplot as plt
import numpy as np
import pyaudio

### WARNING: This code produces a loud, high-pitched tone. Turn down the volume! ###

class SineWave(Thread):
    def __init__(self, amp: float, freq: float, offset: float, time: np.ndarray):
        super().__init__()

        self.signal_lock = Lock()

        self.amp_lock = Lock()
        self._amp = amp
        self.freq_lock = Lock()
        self._freq = freq
        self.offset_lock = Lock()
        self._offset = offset

        self.time_lock = Lock()
        self._time = time

    @property
    def signal(self):
        with self.signal_lock:
            factor = 2 * np.pi * self.freq
            return self.amp * np.sin(factor * self.time + self.offset).astype(np.float32)

    @property
    def amp(self):
        with self.amp_lock:
            return self._amp
    @amp.setter
    def amp(self, value):
        with self.amp_lock:
            self._amp = value

    @property
    def freq(self):
        with self.freq_lock:
            return self._freq
    @freq.setter
    def freq(self, value):
        with self.freq_lock:
            self._freq = value

    @property
    def offset(self):
        with self.offset_lock:
            return self._offset
    @offset.setter
    def offset(self, value):
        with self.offset_lock:
            self._offset = value

    @property
    def time(self):
        with self.time_lock:
            return self._time
    @time.setter
    def time(self, value):
        with self.time_lock:
            self._time = value

    def run(self):
        return self.signal
        pass

class Playback(Thread):
    def __init__(self, rate):
        super().__init__()

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)

        self.signal_lock = Lock()
        self._signal = None

    @property
    def signal(self):
        with self.signal_lock:
            return self._signal
    @signal.setter
    def signal(self, value):
        with self.signal_lock:
            self._signal = value

    def run(self):
        self.stream.start_stream()

    def play(self):
        if self.signal is None:
            pass
        else:
            self.stream.write(self.signal)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        return

class Evaluator(Thread):
    def __init__(self, x, y, z, axis=0, matrix=np.identity(3)):
        super().__init__()

        self.signal_lock = Lock()

        self.transform_matrix_lock = Lock()
        self._matrix = matrix

        self._x = x
        self._y = y
        self._z = z

        self.axis_lock = Lock()
        self.axis = axis

        self.iterator = range(self.length)

    @property
    def length(self):
        return len(self.x.signal)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def matrix(self):
        with self.transform_matrix_lock:
            return self._matrix
    @matrix.setter
    def matrix(self, value):
        with self.transform_matrix_lock:
            self._matrix = value

    @property
    def axis(self):
        with self.axis_lock:
            return self._axis
    @axis.setter
    def axis(self, value):
        with self.axis_lock:
            self._axis = value

    @property
    def signal(self):

        with self.signal_lock:
            # get each element of x.signal, y.signal, z.signal and compose them into an array of vectors
            vector_array = np.array([self.x.signal, self.y.signal, self.z.signal]).T
            vector_array = np.matmul(vector_array, self.matrix)
            result = vector_array[:, self.axis].astype(np.float32)
            return result

    def run(self):
        return self.signal
        pass

rate = 44100
window = 1
time = np.arange(0, window, 1/rate).astype(np.float32)

playback = Playback(rate)
playback.start()

x = SineWave(1, 440, 0, time)
y = SineWave(1, 440, 0, time)
z = SineWave(1, 440, 0, time)
x.start()
y.start()
z.start()

evaluator = Evaluator(x, y, z, axis=0)

evaluator.start()

# evaluator.join()
theta = 0
i = 0

while True:
    theta_x = np.radians(i)
    theta_y = np.radians(i)
    theta_z = np.radians(i)
    matrix_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    matrix = np.matmul(matrix_x, matrix_y)
    matrix = np.matmul(matrix, matrix_z)

    evaluator.matrix = matrix

    x.join()
    y.join()
    z.join()

    evaluator.join()

    playback.signal = evaluator.signal
    playback.play()
    playback.join()

    i += 11.25
