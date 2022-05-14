import { Injectable } from '@nestjs/common';
import { CreateTweetDto } from './dto/create-tweet.dto';
import { Tweet } from './entities/tweet.entity';

@Injectable()
export class TwitterService {
  private tweet_counter = 1;
  private tweets: Tweet[] = [];

  createTweet(newTweet: CreateTweetDto): void {
    this.tweets.push({ tweet_id: this.tweet_counter++, ...newTweet });
  }

  getTweet(tweet_id: number): Tweet {
    const [tweet] = this.tweets.filter((tw) => tw.tweet_id === tweet_id);
    return tweet;
  }

  getAllTweets(): Tweet[] {
    return this.tweets;
  }

  updateTweet(tweet_id: number, newContent: string) {
    const [tweet] = this.tweets.filter((t) => t.tweet_id === tweet_id);
    tweet.content = newContent;
    return;
  }

  deleteTweet(tweet_id: number) {
    this.tweets = this.tweets.filter((t) => t.tweet_id !== tweet_id);
    return;
  }
}
