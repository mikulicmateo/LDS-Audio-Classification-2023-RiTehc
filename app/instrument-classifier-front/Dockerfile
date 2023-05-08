FROM node:20

WORKDIR /frontend

COPY package.json ./
COPY package-lock.json ./

ENV PATH /frontend/node_modules/.bin:$PATH

RUN npm ci --silent
RUN npm install react-scripts@3.4.1 -g --silent

COPY . /frontend/

RUN npm run build
RUN npm install -g serve

CMD serve -s build -l 3006