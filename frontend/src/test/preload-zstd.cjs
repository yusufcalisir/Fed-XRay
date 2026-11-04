// Polyfill zlib.createZstdDecompress for Node <= 20 to prevent undici decompress.js crash
const zlib = require("node:zlib");
if (typeof zlib.createZstdDecompress === "undefined") {
  zlib.createZstdDecompress = function () {
    throw new Error("zstd compression is not natively supported in this Node.js version");
  };
}

