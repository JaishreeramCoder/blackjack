# Project Setup

## Prerequisites
- Node.js installed
- npm installed

## Installation
```sh
npm install
```

## Tailwind CSS Setup
Run the following command to generate and watch the Tailwind CSS output:
```sh
npx @tailwindcss/cli -i ./src/style/input.css -o ./src/style/output.css --watch
```
This will generate `output.css` and update it whenever new Tailwind classes are used.

## Start the Server
```sh
npm start
```

## File Structure
```
/project-root
│── src/
│   ├── style/
│   │   ├── input.css
│   │   ├── output.css
│   ├── components/
│── package.json
│── README.md
```

## Development
- Ensure `npx @tailwindcss/cli --watch` is running for CSS updates.
- Use `npm start` to launch the development server.

## Build for Production
```sh
npm run build

