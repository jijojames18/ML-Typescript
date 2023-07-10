# Machine Learning - Using Typescript

Common machine learning algorithms - implemenented using TypeScript. All algorithms are placed inside the `algorithms` folder. The `implementations` folder contains solutions to various problems using the algorithms.

Use `npm run build` to build the entire project or use `npx tsc <file_name>` to build only a particular file.
All compiled JavaScript files will be listed under `dist` folder.
Use `node --inspect-brk dist/implementations/<algorithm>/<filename>.js` to debug.  

## Customize configuration

## Project Setup

```sh
npm install
```

### Compile all algorithms and implementations

```sh
npm run build
```

### Run a particular implementation

```sh
node dist/implementations/<algorithm>/<filename>.js
```

### Lint with [ESLint](https://eslint.org/)

```sh
npm run lint
```
