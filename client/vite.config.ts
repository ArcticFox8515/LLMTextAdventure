import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:3001',
      '/socket.io': {
        target: 'ws://localhost:3002',
        ws: true
      }
    }
  },
  build: {
    outDir: '../dist/client',
    emptyOutDir: true
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  }
})