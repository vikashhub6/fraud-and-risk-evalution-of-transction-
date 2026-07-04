# --- Stage 1: build the React app ---
FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Bake the backend URL in at build time (Create React App reads env vars
# only during build, not at container runtime).
ARG REACT_APP_API_BASE=http://localhost:8000
ENV REACT_APP_API_BASE=$REACT_APP_API_BASE
RUN npm run build

# --- Stage 2: serve the built static files with nginx ---
# Final image only contains the compiled output + a tiny web server —
# not Node.js, not node_modules. Much smaller than shipping the build stage.
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
