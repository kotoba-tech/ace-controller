# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Kotoba API Service implementation.

For documentation on how to configure the Kotoba ASR API, please refer to the
[Kotoba API Reference](https://docs.kotobatech.ai/).
"""

import asyncio
import base64
import concurrent.futures
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import websockets
from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from nvidia_pipecat.utils.tracing import AttachmentStrategy, traceable, traced


@dataclass
class KotobaASRInternalState:
    """Internal state for VAD simuation."""

    transcript: str = ""
    silence: int = 0
    threshold: int = 10

    def __add__(self, text_chunk: str) -> None:
        if text_chunk is not None:
            self.transcript += text_chunk
            if text_chunk:
                self.silence = 0
            else:
                self.silence += 1
        return self

    @property
    def is_final(self) -> bool:
        """Determine if the transcription is final.

        When the number of consecutive empty chunks reaches the threshold,
        the current transcription is considered final.

        Returns:
            bool: True if the number of consecutive silences exceeds the threshold (10)
        """
        return self.silence >= self.threshold

    def _reset(self) -> None:
        self.transcript = ""
        self.silence = 0

    def finalize(self) -> str:
        """Finalize and return the current transcription text.

        Returns the accumulated transcription text and resets the internal state.
        This method is called when the transcription is determined to be complete
        due to consecutive silence detection.

        Returns:
            str: The final transcription text
        """
        ret = self.transcript
        self._reset()
        return ret


@traceable
class KotobaASRService(STTService):
    """Kotoba Automatic Speech Recognition service using API hosted by Kotoba Technologies, Japan.

    This service provides speech-to-text conversion using Kotoba Technologies'
    WebSocket-based real-time ASR API. It streams audio data through a WebSocket
    connection and receives both interim and final transcription results.

    Key features:
    - Real-time speech recognition
    - Provides interim transcription results, delivered every 80ms
    - Simulated Voice Activity Detection (VAD) by silence duration
    - Automatic transcription finalization through silence detection
    """

    def __init__(
        self,
        *,
        server: str,
        api_key: str | None = None,
        language: Language | None = Language.JA,
        sample_rate: int = 16_000,
        audio_channel_count: int = 1,
        interim_results: bool = True,
        silence_seconds: float = 1.0,
        generate_interruptions: bool = False,  # Only set to True if transport VAD is disabled
        idle_timeout: int = 30,  # Timeout for idle Kotoba ASR request
        **kwargs,
    ):
        """Initializes the Kotoba ASR service.

        Args:
            api_key: Kotoba API key
            language: Language for recognition.
            sample_rate: Audio sample rate in Hz.
            audio_channel_count: Number of audio channels.
            interim_results: Enable interim results.
            silence_seconds: Silence duration to determine final results.
            generate_interruptions: Enable interruption events.
            idle_timeout: Timeout for idle ASR request in seconds.
            use_ssl: Enable SSL connection.
            **kwargs: Additional arguments for STTService.
        """
        super().__init__(**kwargs)
        self._server = server
        self._language_code = language
        self._sample_rate: int = sample_rate
        self._audio_channel_count = audio_channel_count
        self._interim_results = interim_results
        self._idle_timeout = idle_timeout
        self.last_transcript_frame = None
        self._api_key = api_key

        self._queue = asyncio.Queue()
        self._generate_interruptions = generate_interruptions
        if self._generate_interruptions:
            self._vad_state = VADState.QUIET

        # Initialize the thread task and response task
        self._thread_task = None
        self._response_task = None
        self._thread_running = False

        self._state = KotobaASRInternalState(threshold=silence_seconds // 0.080)
        # Initialize websocket connection
        self._client = None
        # Create a task to connect websocket asynchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is already running, create a task
                self._connection_task = asyncio.create_task(self._connect_websocket())
            else:
                # If no event loop is running, run the coroutine
                asyncio.run(self._connect_websocket())
        except RuntimeError:
            # If we can't get an event loop, we'll connect in start() method
            logger.warning("Could not connect WebSocket in __init__, will connect in start() method")
            self._connection_task = None

    async def _connect_websocket(self):
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            logger.info("Connecting to Kotoba API...")
            self._client = await websockets.connect(f"wss://{self._server}/v1/realtime", extra_headers=headers)

            # Send initialization message
            init_message = {
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_number_of_channels": self._audio_channel_count,
                    "input_audio_sample_rate": self._sample_rate,
                    "input_audio_transcription": {
                        "language": str(self._language_code),
                        "target_language": str(self._language_code),
                    },
                },
            }
            await self._client.send(json.dumps(init_message))
            logger.info("Kotoba ASR service connected and initialized")

        except Exception as e:
            logger.error(f"Failed to connect to Kotoba API: {e}")
            raise

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            bool: False as this service does not support metric generation.
        """
        return False

    async def start(self, frame: StartFrame):
        """Start the ASR service.

        Args:
            frame: The StartFrame that triggered the start.
        """
        await super().start(frame)

        # Ensure WebSocket is connected
        if self._connection_task:
            # Wait for connection to complete if it's still in progress
            await self._connection_task
        elif not self._client:
            # Connect if not already connected
            await self._connect_websocket()

        self._response_task = self.create_task(self._response_task_handler())
        self._response_queue = asyncio.Queue()
        logger.info("KotobaASRService started")

    async def stop(self, frame: EndFrame):
        """Stop the ASR service and cleanup resources.

        Args:
            frame: The EndFrame that triggered the stop.
        """
        await super().stop(frame)
        await self._stop_tasks()

    async def cancel(self, frame: CancelFrame):
        """Cancel the ASR service and cleanup resources.

        Args:
            frame: The CancelFrame that triggered the cancellation.
        """
        await super().cancel(frame)
        await self._stop_tasks()

    async def _stop_tasks(self):
        if self._thread_task is not None and not self._thread_task.done():
            await self.cancel_task(self._thread_task)
        if self._response_task is not None and not self._response_task.done():
            await self.cancel_task(self._response_task)
        # Close WebSocket connection
        if self._client:
            await self._client.close()

    def _response_handler(self):
        try:
            for audio_chunk in self:  # requests
                event = {"type": "input_audio_buffer.append", "audio": base64.b64encode(audio_chunk).decode()}
                asyncio.run_coroutine_threadsafe(self._client.send(json.dumps(event)), self.get_event_loop())
        except Exception as e:
            logger.error(f"Error in Kotoba ASR stream: {e}")
            raise

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="asr")
    async def _thread_task_handler(self):
        try:
            self._thread_running = True
            await asyncio.to_thread(self._response_handler)
        except asyncio.CancelledError:
            self._thread_running = False
            raise

    async def _handle_interruptions(self, frame: Frame):
        if self.interruptions_allowed:
            # Make sure we notify about interruptions quickly out-of-band.
            if isinstance(frame, UserStartedSpeakingFrame):
                logger.debug("User started speaking")
                await self._start_interruption()
                # Push an out-of-band frame (i.e. not using the ordered push
                # frame task) to stop everything, specially at the output
                # transport.
                await self.push_frame(StartInterruptionFrame())
            elif isinstance(frame, UserStoppedSpeakingFrame):
                logger.debug("User stopped speaking")
                await self._stop_interruption()
                await self.push_frame(StopInterruptionFrame())

        await self.push_frame(frame)

    async def _handle_response(self, response):
        """Process response from Kotoba API and generate transcription frames."""
        if response.get("type") == "conversation.item.input_audio_transcription.delta":
            transcript = response.get("delta", "")
            self._state += transcript
            await self.stop_ttfb_metrics()
            # Check if this is a final transcript
            if self._state.is_final:
                await self.stop_processing_metrics()
                if self._generate_interruptions:
                    self._vad_state = VADState.QUIET
                    await self._handle_interruptions(UserStoppedSpeakingFrame())
                finalized_transcript = self._state.finalize()
                if len(finalized_transcript) > 0:  # Long silence and no transcription
                    logger.debug(f"Final user transcript: [{finalized_transcript}]")
                    frame = TranscriptionFrame(finalized_transcript, "", time_now_iso8601(), None)
                    await self.push_frame(frame)
                    self.last_transcript_frame = None
            elif self._interim_results:
                # Handle interim results
                if self._generate_interruptions and self._vad_state != VADState.SPEAKING:
                    self._vad_state = VADState.SPEAKING
                    await self._handle_interruptions(UserStartedSpeakingFrame())

                logger.debug(f"Interim user transcript: [{transcript}]")
                frame = InterimTranscriptionFrame(transcript, "", time_now_iso8601(), None)
                await self.push_frame(frame)
                self.last_transcript_frame = frame

    async def _response_task_handler(self):
        while True:
            try:
                response = await self._client.recv()
                # Parse JSON response and handle it
                if response:
                    data = json.loads(response)
                    await self._handle_response(data)
            except asyncio.CancelledError:
                break
            except websockets.exceptions.ConnectionClosed:
                logger.error("WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error in response handler: {e}")
                break

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run speech-to-text recognition.

        Args:
            audio: The audio data to process.

        Yields:
            Frame: A sequence of frames containing the recognition results.
        """
        if self._thread_task is None or self._thread_task.done():
            self._thread_task = self.create_task(self._thread_task_handler())
        await self._queue.put(audio)
        yield None

    def __next__(self) -> bytes:
        """Get the next audio chunk for processing.

        Returns:
            bytes: The next audio chunk.

        Raises:
            StopIteration: When no more audio chunks are available.
        """
        if not self._thread_running:
            raise StopIteration
        try:
            future = asyncio.run_coroutine_threadsafe(self._queue.get(), self.get_event_loop())
            result = future.result(timeout=self._idle_timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.info(
                f"ASR service is idle for {self._idle_timeout} seconds, terminating active Kotoba ASR request..."
            )
            self._thread_task = None
            raise StopIteration from None
        except Exception as e:
            future.cancel()
            raise e
        return result

    def __iter__(self):
        """Get iterator for audio chunks.

        Returns:
            KotobaASRService: Self reference for iteration.
        """
        return self
