from threading import Thread, Lock
import matplotlib.pyplot as plt
import numpy as np
import pyaudio

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

        self.x_lock = Lock()
        self._x = x

        self.y_lock = Lock()
        self._y = y

        self.z_lock = Lock()
        self._z = z

        self.axis_lock = Lock()
        self.axis = axis

        self.iterator = range(self.length)

    @property
    def length(self):
        return len(self.x.signal)

    @property
    def x(self):
        with self.x_lock:
            return self._x

    @property
    def y(self):
        with self.y_lock:
            return self._y

    @property
    def z(self):
        with self.z_lock:
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
window = 0.05
time = np.arange(0, window, 1/rate).astype(np.float32)

playback = Playback(rate)
playback.start()

x = SineWave(1, 440, 0, time)
y = SineWave(1, 80, 0, time)
z = SineWave(1, 60, 0, time)
x.start()
y.start()
z.start()

evaluator = Evaluator(x, y, z, axis=1)

evaluator.start()

# evaluator.join()
theta = 0
i = 0

while True:
    theta += 1
    x.freq += 1
    y.freq += 1
    z.freq += 1
    print(x.freq, y.freq, z.freq)
    evaluator.matrix = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

    evaluator.join()

    x.join()
    y.join()
    z.join()

    playback.signal = evaluator.signal
    playback.play()
    playback.join()

