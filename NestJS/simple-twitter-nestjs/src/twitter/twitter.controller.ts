import {
  Body,
  Controller,
  Get,
  Param,
  Post,
  Redirect,
  Render
} from '@nestjs/common';
import { CreateTweetDto } from './dto/create-tweet.dto';
import { TwitterService } from './twitter.service';

@Controller('twitter')
export class TwitterController {
  constructor(private readonly twitterService: TwitterService) {}

  @Get('/')
  @Render('home')
  getHome() {
    const allTweets = this.twitterService.getAllTweets();
    return { allTweets, title: 'All Tweets' };
  }
  @Post('/')
  @Redirect('/twitter')
  postHome(@Body() newTweet: CreateTweetDto) {
    this.twitterService.createTweet(newTweet);
    return;
  }

  @Get('/edit/:tweet_id')
  @Render('edit')
  getEditTweet(@Param('tweet_id') tweet_id: number) {
    const tweet = this.twitterService.getTweet(tweet_id);
    return { tweet_id, content: tweet.content };
  }
  @Post('/edit/:tweet_id')
  @Redirect('/twitter')
  postEditTweet(@Param('tweet_id') tweet_id: number, @Body() { newContent }) {
    this.twitterService.updateTweet(tweet_id, newContent);
    return;
  }

  @Get('/delete/:tweet_id')
  @Redirect('/twitter')
  deleteTweet(@Param('tweet_id') tweet_id: number) {
    this.twitterService.deleteTweet(tweet_id);
    return;
  }
}
