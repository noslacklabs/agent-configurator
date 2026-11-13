import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `npm run dev` serves the page on 5175 and talks to the hand-off service on 8003.
// `npm run build` writes dist/, which the hand-off service serves itself.
export default defineConfig({
  plugins: [react()],
  server: { host: '127.0.0.1', port: 5175, strictPort: true },
})
