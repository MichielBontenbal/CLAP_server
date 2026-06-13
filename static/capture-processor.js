// AudioWorklet that forwards raw mic samples (mono, Float32) to the main thread.
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) {
      // Copy: the underlying buffer is reused by the audio engine.
      this.port.postMessage(new Float32Array(channel));
    }
    return true;
  }
}

registerProcessor('capture', CaptureProcessor);
