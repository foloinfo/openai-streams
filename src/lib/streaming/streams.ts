import { ENCODER, DECODER } from "../../globs/shared";
import { ChatParser, TokenParser } from "./transforms";

import { Transform, pipeline, yieldStream } from "yield-stream";
import { yieldStream as yieldStreamNode } from "yield-stream/node";
import { OpenAIError } from "../errors";

export type StreamMode = "raw" | "tokens";

export interface OpenAIStreamOptions {
  /**
   * Whether to return tokens or raw events.
   */
  mode?: StreamMode;

  /**
   * A function to run at the end of a stream. This is useful if you want
   * to do something with the stream after it's done, like log token usage.
   */
  onDone?: () => void | Promise<void>;
  /**
   * A function that runs for each token. This is useful if you want
   * to sum tokens used as they're returned.
   */
  onParse?: (token: string) => void | Promise<void>;
}

export type OpenAIStream = (
  stream: NodeJS.ReadableStream | ReadableStream<Uint8Array>,
  options: OpenAIStreamOptions
) => ReadableStream<Uint8Array>;

const closeController = async(
  controller: ReadableStreamDefaultController,
  onDone?: () => void | Promise<void>
) => {
  if (controller.desiredSize) {
    controller.close();
  }
  await onDone?.();
};

const bufferIsDone = (buffer: string) => {
  return buffer.startsWith("data: [DONE]") || buffer === "[DONE]";
};

/**
 * A `ReadableStream` of server sent events from the given OpenAI API stream.
 *
 * @note This can't be done via a generator while using `createParser` because
 * there is no way to yield from within the callback.
 */
export const EventStream: OpenAIStream = (
  stream,
  { mode = "tokens", onDone }
) => {
  return new ReadableStream<Uint8Array>({
    async start(controller) {

      // Check if the stream is a NodeJS stream or a browser stream.
      // @ts-ignore - TS doesn't know about `pipe` on streams.
      const isNodeJsStream = typeof stream.pipe === "function";
      let buffer = "";

      for await (const chunk of isNodeJsStream
        ? yieldStreamNode<Buffer>(stream as NodeJS.ReadableStream)
        : yieldStream(stream as ReadableStream<Uint8Array>)
      ) {
        buffer += DECODER.decode(chunk, { stream: true });
        if(bufferIsDone(buffer)){
          await closeController(controller, onDone);
          return;
        }

        while (true) {
          const boundary = buffer.indexOf("\n\n");
          if (boundary === -1) {
            if(bufferIsDone(buffer)){
              await closeController(controller, onDone);
              return;
            }
            break;
          }

          const jsonString = buffer.slice(5, boundary).trim();
          buffer = buffer.slice(boundary + 2);
          if(bufferIsDone(buffer)){
            await closeController(controller, onDone);
            return;
          }

          let parsed = null;
          try {
            parsed = JSON.parse(jsonString);
          } catch (err) {
            // wait for more data to come untill it is a valid json.
          }

          if(typeof parsed === "object"){
            /**
             * Break if choice.finish_reason is "stop".
             * This is for Azure API did not close with [DONE]
             */
            if (parsed?.choices) {
              const { choices } = parsed;
              for (const choice of choices) {
                if(choice?.finish_reason === "stop") {
                  await closeController(controller, onDone);
                  return;
                }
                if (mode === "tokens" && choice?.finish_reason === "length") {
                  throw new OpenAIError("MAX_TOKENS");
                }
              }
            }

            if (parsed.hasOwnProperty("error")) {
              controller.error(new Error(parsed.error.message));
            }

            // only send valid json
            controller.enqueue(ENCODER.encode(jsonString));
            parsed = null;
          }
        }
      }

    },
  });
};

/**
 * Creates a handler that decodes the given stream into a string,
 * then pipes that string into the provided callback.
 */
const CallbackHandler = ({ onParse }: OpenAIStreamOptions) => {
  const handler: Transform  = async function* (chunk) {
    const decoded = DECODER.decode(chunk, { stream: true });
    onParse?.(decoded);
    if (decoded) {
      yield ENCODER.encode(decoded);
    }
  };

  return handler;
};

/**
 * A `ReadableStream` of parsed tokens from the given OpenAI API stream.
 */
export const TokenStream: OpenAIStream = (
  stream,
  options = { mode: "tokens" }
) => {
  return pipeline(
    EventStream(stream, options),
    TokenParser,
    CallbackHandler(options)
  );
};

/**
 * A `ReadableStream` of parsed deltas from the given ChatGPT stream.
 */
export const ChatStream: OpenAIStream = (
  stream,
  options = { mode: "tokens" }
) => {
  return pipeline(
    EventStream(stream, options),
    ChatParser,
    CallbackHandler(options)
  );
};
