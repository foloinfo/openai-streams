/* eslint-disable no-console */
import type NodeFetch from "node-fetch";
import type { RequestInfo as NodeFetchRequestInfo, RequestInit as NodeFetchRequestInit } from "node-fetch";

type Fetch = typeof NodeFetch | typeof fetch;

export interface BackoffOptions {
  maxRetries?: number;
  delay?: number;
  onRateLimitReached?: (
    args: {
      retries: number;
      delay: number;
    }
  ) => void;
}

export const fetchWithBackoff = async (
  input: RequestInfo & NodeFetchRequestInfo,
  init?: RequestInit & NodeFetchRequestInit,
  fetch: Fetch = globalThis.fetch,
  backoffOptions: BackoffOptions = { }
) => {
  if (!fetch) {
    throw new Error("No fetch implementation found.");
  }

  const {
    maxRetries = 7,
    onRateLimitReached = () => null,
  } = backoffOptions;
  let { delay = 500 } = backoffOptions;

  for (let i = 0; i <= maxRetries; i++) {
    try {
      const response = await fetch(input, init);

      if (!response.ok) {
        const errorData = await response.json();
        if (errorData.type === "RATE_LIMIT_REACHED" || errorData?.error?.code === "429") {
          throw new Error("RATE_LIMIT_REACHED");
        }
      }

      return response;
    } catch (error: any) {
      if (
        error.message === "RATE_LIMIT_REACHED" &&
        i < maxRetries
      ) {
        console.log("Rate limit reached. Retrying in " + delay + "ms");
        await onRateLimitReached({ retries: i, delay });

        await new Promise((resolve) => setTimeout(resolve, delay));

        delay *= 2;
      } else {
        throw error;
      }
    }
  }

  throw new Error("Max retries reached.");
};
