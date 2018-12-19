const MongoClient = require('mongodb').MongoClient;
const fs = require('fs');

const url = 'mongodb://localhost:27017';


function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
}


const run = async () => {
    const client = await MongoClient.connect(url, {useNewUrlParser: true, connectTimeoutMS: 0});
    console.log("Connected successfully to server.");

    const db = client.db('namuwiki');
    const rawdata = db.collection('wiki');
    const cursor = rawdata.find({});
    const count = await rawdata.countDocuments();

    let counter = 0, i = 0;
    // download data

    while (await cursor.hasNext()) {
        counter++;
        const doc = await cursor.next();
        const text = doc.text.trim();
        if (text.length < 40) continue;
        //console.log('counter: ' + counter  );

        if (counter % getRandomInt(4000) === 0) {
            i++;
            const file = fs.createWriteStream('WIKI_DATA/' + i + '.txt');
            console.log('WIKI_DATA/' + i + '.txt');
            file.write(text);
            file.close();
        }
        //console.log(doc.text);


    }

    //
    // for (let i = 0; i < cursor.length(); i++) {
    //
    //     let text = cursor[i].text;
    //     if( text.length> 50)
    //     {   max++;
    //         const file = fs.createWriteStream('WIKI_DATA/' + i + '.txt');
    //         console.log('WIKI_DATA/' + i + '.txt');
    //         file.write(text);
    //         file.close();
    //
    //         if(i % 2 === 0) {
    //             continue;
    //         }
    //
    //
    //     }
    //
    // };
    client.close();


};

run().catch((err) => {
    console.log(err);
});